#ifndef HDF5PLOTFILEWRITER_H_
#define HDF5PLOTFILEWRITER_H_

#include "PlotfileWriter.H"

class HDF5PlotfileWriter : public PlotfileWriter
{
    public:
        HDF5PlotfileWriter(
            const std::string& prefix = "plt",
            const std::string& compression = "None@0"
        ) {}

        void Write(
            const AmrMeshState& state,
            int iteration,
            double time
        ) override;

    private:
        std::string m_prefix;
        std::string m_compression;
};

#endif