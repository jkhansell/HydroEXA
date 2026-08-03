#include <state/AmrMeshState.H>
#include <io/CheckpointerContext.H>


void AmrMeshState::GetTerrainData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime) {
    data.clear();
    datatime.clear();
    data.push_back(&DynamicTerrain[lev]);
    datatime.push_back(t_new[lev]);
}

void AmrMeshState::GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime) {
    data.clear();
    datatime.clear();

    const amrex::Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
        data.push_back(&U_new[lev]);
        datatime.push_back(t_new[lev]);
    }
    else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
        data.push_back(&U_old[lev]);
        datatime.push_back(t_old[lev]);
    }
    else
    {
        data.push_back(&U_old[lev]);
        data.push_back(&U_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }
}

void AmrMeshState::WritePlotfile(int iteration, amrex::Real time) {
    IO->WritePlotfile(
        U_new, DynamicTerrain,
        iteration, time,
        Geom(), refRatio(),
        finest_level
    );
}

void AmrMeshState::WriteCheckpoint() {
    IO->WriteCheckpoint(GetCheckpointerContext(), istep[0]);
}


SolverContext AmrMeshState::GetSolverContext()
{
    return
    {

        //==========================================================================
        // Simulation state
        //==========================================================================
        
        U_new,
        U_old,
        DynamicTerrain,

        //==========================================================================
        // Mesh hierarchy
        //==========================================================================

        Geom(),
        grids,
        dmap,
        [this](int lev) { return refRatio(lev); },

        //==========================================================================
        // Boundary conditions
        //==========================================================================

        U_bcs,
        Terrain_bcs,

        //==========================================================================
        // Refluxing
        //==========================================================================

        flux_reg,

        amr_p.do_reflux,
        finest_level,

        //==========================================================================
        // Mesh services
        //==========================================================================

        [this](
            int lev,
            amrex::Real time,
            amrex::MultiFab& mf,
            const amrex::Vector<amrex::BCRec>& bc,
            int icomp,
            int ncomp)
        {
            FillPatch(lev,time,mf,bc,icomp,ncomp);
        },

        [](const amrex::MultiFab& mf,
           const amrex::BoxArray& ba,
           const amrex::IntVect& rr,
           int ngrow,
           int blocking)
        {
            return makeFineMask(mf, ba, rr, ngrow, blocking);
        }
    };
}

CheckpointerContext AmrMeshState::GetCheckpointerContext()
{
    return {
        U_new, U_old, DynamicTerrain,
        grids, dmap,
        dt, t_new, t_old,
        finest_level, istep[0], istep
    };
}