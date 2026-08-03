#ifndef LOGGING_H
#define LOGGING_H

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX.H>
#include <string>

enum class LogLevel {
    INFO  = 0,   // Geometry validation, data integrity, timing, I/O
    DIAG  = 1,   // Pruning results, regrid events, per-level details
    DEBUG = 2,   // Detailed internal state
    EXPL  = 3,   // Numerical method internals, solver details
    WARN  = 4,   // Always prints (warning)
    ERROR = 5    // Always prints + aborts (error)
};

class Logger {
public:
    static Logger& instance() {
        static Logger s_instance;
        return s_instance;
    }

    int verbosity() const { return m_verbosity; }

    void setVerbosity(int v) { m_verbosity = v; }

    bool shouldPrint(LogLevel level) const {
        // WARN and ERROR always print; INFO/DIAG/DEBUG/EXPL respect verbosity
        return static_cast<int>(level) >= static_cast<int>(LogLevel::WARN)
            || static_cast<int>(level) <= m_verbosity;
    }

    void print(LogLevel level, const std::string& msg) const {
        if (amrex::ParallelDescriptor::IOProcessor() && shouldPrint(level)) {
            amrex::Print() << msg;
        }
    }

    void error(const std::string& msg) const {
        amrex::Error(msg);
    }

public:
    Logger() : m_verbosity(0) {
        amrex::ParmParse pp("HydroEXA");
        pp.query("log_verbosity", m_verbosity);
    }
    int m_verbosity;
};

#define LOG(level, msg)  Logger::instance().print(LogLevel::level, msg)
#define LOG_WARN(msg)    Logger::instance().print(LogLevel::WARN, std::string("[WARNING] ") + msg)
#define LOG_ERROR(msg)   Logger::instance().error(std::string("[ERROR] ") + msg)

#endif // LOGGING_H
