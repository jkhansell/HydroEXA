#ifndef PLOTFILEWRITER_H_
#define PLOTFILEWRITER_H_

#include <string>
class AmrMeshState;

class PlotfileWriter
{
    public:
        PlotfileWriter() = default;
        virtual ~PlotfileWriter() = default;
        virtual void Write(const AmrMeshState& state, int iteration, double time);
};

#endif