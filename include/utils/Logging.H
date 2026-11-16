#ifndef LOGGING_H
#define LOGGING_H

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX.H>
#include <string>

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
 * Usage: ``LOG(INFO, "message")``, ``LOG_WARN("...")``, ``LOG_ERROR("...")``.
 * Verbosity comes from ``HydroEXA.log_verbosity`` in the inputs file.
 */

/**
 * \brief Severity of a log record, ordered from most to least common.
 *
 * The numeric order is what makes ``shouldPrint`` a one-line comparison:
 * levels below WARN are filtered by verbosity, WARN and ERROR always print.
 */
enum class LogLevel {
    INFO  = 0,   ///< Geometry validation, data integrity, timing, I/O.
    DIAG  = 1,   ///< Pruning results, regrid events, per-level details.
    DEBUG = 2,   ///< Detailed internal state.
    EXPL  = 3,   ///< Numerical-method internals, solver details.
    WARN  = 4,   ///< Always prints, prefixed [WARNING].
    ERROR = 5    ///< Always prints, prefixed [ERROR], then aborts.
};

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
    int verbosity() const { return m_verbosity; }

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
        // WARN and ERROR always print; INFO/DIAG/DEBUG/EXPL respect verbosity
        return static_cast<int>(level) >= static_cast<int>(LogLevel::WARN)
            || static_cast<int>(level) <= m_verbosity;
    }

    /**
     * \brief Emit \p msg on the IO processor when \p level is enabled.
     * \param level Severity of the record.
     * \param msg Text to write, without a trailing newline.
     */
    void print(LogLevel level, const std::string& msg) const {
        if (amrex::ParallelDescriptor::IOProcessor() && shouldPrint(level)) {
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
        pp.query("log_verbosity", m_verbosity);
    }

private:
    int m_verbosity;  ///< Threshold from ``HydroEXA.log_verbosity``.
};

/// Emit \p msg at \p level through the singleton logger.
#define LOG(level, msg)  Logger::instance().print(LogLevel::level, msg)
/// Emit a [WARNING] record; always printed.
#define LOG_WARN(msg)    Logger::instance().print(LogLevel::WARN, std::string("[WARNING] ") + msg)
/// Emit an [ERROR] record and abort the run.
#define LOG_ERROR(msg)   Logger::instance().error(std::string("[ERROR] ") + msg)

#endif // LOGGING_H
