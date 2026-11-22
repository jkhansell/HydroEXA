#ifndef LOGGING_H
#define LOGGING_H

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX.H>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

/**
 * \file Logging.H
 * \brief Structured logging for the solver, gated by input-file verbosity.
 *
 * Two things make this different from sprinkling amrex::Print() around:
 *
 * - **a level**, so a production run prints warnings and a development run can
 *   print solver internals without editing code; and
 * - **a single IO rank**, so a 64-rank run does not print the same line 64
 *   times.
 *
 * Usage: ``LOG(INFO, "message")``, ``LOG_WARN("...")``, ``LOG_ERROR("...")``,
 * ``LOG_STREAM(INFO) << "..."``.
 * Verbosity comes from ``HydroEXA.log_verbosity`` in the inputs file.
 */

/**
 * \brief Severity of a log record, ordered from most to least common.
 *
 * The numeric order is what makes ``shouldPrint`` a one-line comparison:
 * levels below WARN are filtered by verbosity, WARN and ERROR always print.
 */
enum class LogLevel {
    QUIET = -1,  ///< Nothing but WARN/ERROR (batch runs).
    INFO  =  0,  ///< Geometry validation, data integrity, timing, I/O.
    DIAG  =  1,  ///< Pruning results, regrid events, per-level details.
    DEBUG =  2,  ///< Detailed internal state.
    EXPL  =  3,  ///< Numerical-method internals, solver details.
    WARN  =  4,  ///< Always prints, prefixed [WARNING].
    ERROR =  5   ///< Always prints, prefixed [ERROR], then aborts.
};

// ============================================================================
// Null stream — compiles << chains but discards everything
// ============================================================================

namespace detail {
class NullStreamBuf : public std::streambuf {
    int overflow(int c) override { return c; }
    int_type underflow() override { return traits_type::eof(); }
    int_type uflow()   override { return traits_type::eof(); }
};

class NullStream : public std::ostream {
    NullStreamBuf buf;
public:
    NullStream() : std::ostream(&buf) {}
};
} // namespace detail

// ============================================================================
// Logger singleton — reads verbosity on first use
// ============================================================================

/**
 * \brief Process-wide logger.
 *
 * A Meyers singleton: constructing it reads the inputs file, so the first
 * log call is also the first moment verbosity is known.  Only the IO
 * processor writes, so call sites need no rank guard.
 */
class Logger {
public:
    /**
     * \brief The single process-wide instance.
     * \return Reference to the logger, constructed on first use.
     */
    static Logger& instance() {
        static Logger s_instance;
        return s_instance;
    }

    /**
     * \brief Current verbosity threshold.
     * \return Verbosity as set by ``HydroEXA.log_verbosity`` (0 when unset).
     */
    static int verbosity() { return instance().m_verbosity; }

    /**
     * \brief Override the verbosity threshold.
     * \param v New threshold, matching the LogLevel numeric values.
     */
    void setVerbosity(int v) { m_verbosity = v; }

    /**
     * \brief Decide whether a record at \p level is emitted.
     * \param level Severity of the record.
     * \return True when the record passes the verbosity threshold, and for
     *         WARN/ERROR unconditionally.
     */
    bool shouldPrint(LogLevel level) const {
        return static_cast<int>(level) >= static_cast<int>(LogLevel::WARN)
            || static_cast<int>(level) <= m_verbosity;
    }

    /**
     * \brief Emit \p msg on the IO processor when \p level is enabled.
     * \param level Severity of the record.
     * \param msg Text to write, without a trailing newline.
     */
    void print(LogLevel level, const std::string& msg) const {
        if (amrex::ParallelContext::IOProcessorSub() && shouldPrint(level)) {
            amrex::Print() << msg;
        }
    }

    /**
     * \brief Abort the run with \p msg.
     * \param msg Text passed to amrex::Error.
     */
    void error(const std::string& msg) const {
        amrex::Error(msg);
    }

    /**
     * \brief Construct the logger and read ``HydroEXA.log_verbosity``.
     *
     * Reads the inputs file, so it must not be called before
     * ``amrex::Initialize`` has parsed the command line.
     */
    Logger() : m_verbosity(0) {
        amrex::ParmParse pp("HydroEXA");
        std::string lvl;
        if (pp.query("log_verbosity", lvl)) {
            m_verbosity = parse_level(lvl);
        } else if (int v; pp.query("log_verbosity", v)) {
            m_verbosity = v;
        } else if (bool vb; pp.query("verbose", vb)) {
            // Deprecated alias: HydroEXA.verbose
            m_verbosity = vb ? 0 : -1;
            // Use amrex::ErrorStream directly since LOG_WARN isn't defined yet
            if (amrex::ParallelContext::IOProcessorSub()) {
                amrex::ErrorStream() << "[WARNING] HydroEXA.verbose is deprecated; use HydroEXA.log_verbosity\n";
                amrex::ErrorStream().flush();
            }
        }
    }

    int m_verbosity;  ///< Threshold from ``HydroEXA.log_verbosity``.

private:
    /**
     * \brief Parse a verbosity level from a string.
     * \param s Input string (case-insensitive).
     * \return Numeric LogLevel value.
     * \throws amrex::Abort on unknown level name.
     */
    static int parse_level(const std::string& s) {
        std::string upper;
        upper.reserve(s.size());
        for (char c : s) upper += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

        if (upper == "QUIET" || upper == "-1") return -1;
        if (upper == "INFO"  || upper == "0")  return  0;
        if (upper == "DIAG"  || upper == "1")  return  1;
        if (upper == "DEBUG" || upper == "2")  return  2;
        if (upper == "EXPL"  || upper == "3")  return  3;
        if (upper == "WARN"  || upper == "4")  return  4;
        if (upper == "ERROR" || upper == "5")  return  5;

        amrex::Abort("Unknown HydroEXA.log_verbosity value \"" + s +
                     "\". Valid values: QUIET(-1), INFO(0), DIAG(1), "
                     "DEBUG(2), EXPL(3), WARN(4), ERROR(5).");
        return 0; // unreachable
    }
};

// ============================================================================
// LogRecord — lazy, stream-based record
// ============================================================================

/**
 * \brief A single log record that formats lazily.
 *
 * When the record's level is filtered, ``stream()`` returns a null ostream
 * so ``<<`` chains still compile and cost one virtual call per << — no
 * string allocation.  When the level passes, formatting goes to the real
 * stream.  The destructor appends exactly one ``'\n'``.
 *
 * WARN and ERROR route to ``amrex::ErrorStream()`` (stderr); everything else
 * routes to ``amrex::OutStream()`` (stdout), IO-processor only.
 */
class LogRecord {
public:
    LogRecord(LogLevel level)
        : m_level(level)
    {
        const bool is_io = amrex::ParallelContext::IOProcessorSub();
        const bool passes = static_cast<int>(level) >= static_cast<int>(LogLevel::WARN)
                         || static_cast<int>(level) <= Logger::verbosity();

        if (is_io && passes) {
            if (static_cast<int>(level) >= static_cast<int>(LogLevel::WARN)) {
                m_os = &amrex::ErrorStream();
            } else {
                m_os = &amrex::OutStream();
            }
            if (m_os) m_os->precision(10);
        } else {
            m_os = nullptr;
        }
    }

    ~LogRecord() {
        if (m_os) {
            (*m_os) << '\n';
            m_os->flush();
        }
    }

    // Prevent copying
    LogRecord(const LogRecord&) = delete;
    LogRecord& operator=(const LogRecord&) = delete;

    std::ostream& stream() {
        if (m_os) return *m_os;
        static detail::NullStream null_stream;
        return null_stream;
    }

private:
    LogLevel   m_level;
    std::ostream* m_os;  // &OutStream / &ErrorStream / nullptr
};

/// Emit a stream-based record at \p level.  When filtered, << still compiles.
#define LOG_STREAM(level)  ::LogRecord(LogLevel::level).stream()

/// Emit a stream-based record only when \p cond is true.
#define LOG_IF(level, cond) if (cond) LOG_STREAM(level)

/// Emit \p msg at \p level through the singleton logger.
#define LOG(level, msg)  Logger::instance().print(LogLevel::level, msg)
/// Emit a [WARNING] record; always printed to stderr.
#define LOG_WARN(msg)    Logger::instance().print(LogLevel::WARN, std::string("[WARNING] ") + msg)
/// Emit an [ERROR] record and abort the run.
#define LOG_ERROR(msg)   Logger::instance().error(std::string("[ERROR] ") + msg)

#endif // LOGGING_H
