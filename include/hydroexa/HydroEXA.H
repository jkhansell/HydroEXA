#ifndef HydroEXA_H_
#define HydroEXA_H_

#include <AMReX_AmrLevel.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_LayoutData.H>

#include <hydroexa/HydroEXA_prob_parm.H>

using namespace amrex;
class HydroEXA
    :
    public amrex::AmrLevel
{
public:

    HydroEXA ();
    HydroEXA (amrex::Amr&            papa,
         int                    lev,
         const amrex::Geometry& level_geom,
         const amrex::BoxArray& bl,
         const amrex::DistributionMapping& dm,
         amrex::Real            time);
    virtual ~HydroEXA ();

    HydroEXA (const HydroEXA& rhs) = delete;
    HydroEXA& operator= (const HydroEXA& rhs) = delete;

    // Restart from a checkpoint file.
    virtual void restart (amrex::Amr&     papa,
                          std::istream&   is,
                          bool            bReadSpecial = false) override;

    // Write checkpoint
    virtual void checkPoint(const std::string& dir,
                            std::ostream&      os,
                            amrex::VisMF::How  how = amrex::VisMF::NFiles,
                            bool               dump_old = true) override;

    virtual std::string thePlotFileType () const override {
        return {"SERGHEI-AMReX-v0.1"};
    }

    // Write a plotfile to specified directory.
    virtual void writePlotFile (const std::string& dir,
                                std::ostream&      os,
                                amrex::VisMF::How  how) override;

    // Initialize data on this level from another HydroEXA (during regrid).
    virtual void init (amrex::AmrLevel& old) override;

    // Initialize data on this level after regridding if old level did not previously exist
    virtual void init () override;

    // Initialize grid data at problem start-up.
    virtual void initData () override;

    // Advance grids at this level in time.
    virtual amrex::Real advance (amrex::Real time,
                                 amrex::Real dt,
                                 int  iteration,
                                 int  ncycle) override;

    virtual void computeInitialDt (int                                  finest_level,
                                   int                                  sub_cycle,
                                   amrex::Vector<int>&                  n_cycle,
                                   const amrex::Vector<amrex::IntVect>& ref_ratio,
                                   amrex::Vector<amrex::Real>&          dt_level,
                                   amrex::Real                          stop_time) override;

    virtual void computeNewDt (int                                  finest_level,
                               int                                  sub_cycle,
                               amrex::Vector<int>&                  n_cycle,
                               const amrex::Vector<amrex::IntVect>& ref_ratio,
                               amrex::Vector<amrex::Real>&          dt_min,
                               amrex::Vector<amrex::Real>&          dt_level,
                               amrex::Real                          stop_time,
                               int                                  post_regrid_flag) override;

    virtual void post_regrid (int lbase, int new_finest) override;

    // Do work after timestep().
    virtual void post_timestep (int iteration) override;

    // After a full time step
    virtual void postCoarseTimeStep (amrex::Real time) override;

    // Do work after init().
    virtual void post_init (amrex::Real stop_time) override;

    virtual void post_restart () override;

    // Error estimation for regridding.
    virtual void errorEst (amrex::TagBoxArray& tb,
                           int                 clearval,
                           int                 tagval,
                           amrex::Real         time,
                           int                 n_error_buf = 0,
                           int                 ngrow = 0) override;

//    virtual int WorkEstType () override { return Cost_Type; }

    // Define data descriptors.
    static void variableSetUp ();

    // Cleanup data descriptors at end of run.
    static void variableCleanUp ();

    static int numGrow() { return NUM_GROW; }

#if !defined(AMREX_USE_CUDA)
protected:
#endif

    static void read_params ();

    HydroEXA& getLevel (int lev) { return dynamic_cast<HydroEXA&>(parent->getLevel(lev)); }

    void avgDown ();

    void buildMetrics ();

    amrex::Real estTimeStep ();

    // Compute initial time step.
    amrex::Real initialTimeStep ();

    void compute_amrex_effective_fluxes(
        const amrex::Box& face_box,
        amrex::Array4<amrex::Real const> const& U,
        amrex::Array4<amrex::Real const> const& terrain,
        amrex::Array4<amrex::Real> const& D_minus_mf,
        amrex::Array4<amrex::Real> const& D_plus_mf,
        const amrex::Real dt,
        const amrex::Real dx,
        const int dir);


    void printTotal () const;

    std::unique_ptr<amrex::FluxRegister> flux_reg;

    static constexpr int NUM_GROW = 1;
    
    enum StateVariable {
        H = 0, HU, HV,
        NUM_STATE
    };

    enum TerrainVariables {
        Z = 0, 
        R, 
        NUM_TERRAIN
    };

    enum StateDataType {
        State_Type = 0,
        Terrain_Type, 
        NUM_STATE_DATA_TYPE
    };
    static int num_state_data_types;

    static amrex::BCRec phys_bc;

    // Parameters
    static int verbose;
    static amrex::IntVect hydro_tile_size;
    static amrex::Real cfl;

    static int do_reflux;

    static int refine_max_h_grad_lev;
    static int refine_max_umag_grad_lev;

    static amrex::Real h_grad_threshold;
    static amrex::Real umag_grad_threshold;

};

void HydroEXA_bcfill (amrex::Box const& bx, amrex::FArrayBox& data,
                 const int dcomp, const int numcomp,
                 amrex::Geometry const& geom, const amrex::Real time,
                 const amrex::Vector<amrex::BCRec>& bcr, const int bcomp,
                 const int scomp);

AMREX_GPU_HOST_DEVICE
void
hydroexa_tagging (int i, int j, int k,
             amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             amrex::Real h_grad_thresh, 
             amrex::Real u_grad_thresh, 
             char tagval);

#endif
