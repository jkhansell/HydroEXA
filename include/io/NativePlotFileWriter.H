#ifndef NATIVEPLOTFILEWRITER_H_
#define NATIVEPLOTFILEWRITER_H_

#include "PlotfileWriter.H"

class NativePlotfileWriter : public PlotfileWriter
{
public:
    explicit NativePlotfileWriter(
        const std::string& prefix = "plt");

    void Write(
        const AmrMeshState& state,
        int iteration,
        double time) override;

private:
    std::string m_prefix;
};

#endif