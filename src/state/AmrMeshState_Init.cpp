#include <state/AmrMeshState.H>
#include <utils/Logging.H>



AmrMeshState::AmrMeshState(std::shared_ptr<IOHandler> io_handler,
                           int terrain_ref_lev, 
                           HDF5SpatialMetadata& meta,
                           std::string input,
                           RuntimeParameters runtime_params,
                           IOParameters io_params,
                           AMRParameters amr_params,
                           PhysicsParameters physics_params,
                           /*AmrCore params*/
                           const amrex::RealBox& rb, 
                           int max_level_in,
                           const amrex::Vector<int>& n_cell_in, 
                           int coord,
                           const amrex::Vector<amrex::IntVect>& ref_ratios)
    : amrex::AmrCore(&rb, max_level_in, n_cell_in, coord, ref_ratios),
      IO(io_handler),
      static_terrain_lev(terrain_ref_lev), 
      metadata(meta), 
      hdf5_path(input),
      ncomp_U(3),
      ngrow_U(1),
      ncomp_Terrain(2),
      ngrow_Terrain(1), 
      runtime_p(runtime_params), 
      io_p(io_params), 
      amr_p(amr_params), 
      physics_p(physics_params) 
{
    // 2. Resize containers dynamically
    ResizeLevels(max_level+1); // Resizes all variables

    if (amr_p.do_subcycle) {
        for (int lev = 1; lev <= max_level; ++lev) {
            nsubsteps[lev] = MaxRefRatio(lev-1);
        }
    }


    // 3. Populate fluid boundary conditions record tracking

    const auto& period = geom[0].periodicity();
    
    U_bcs.resize(ncomp_U);
    Terrain_bcs.resize(ncomp_Terrain);

    for (int comp = 0; comp < ncomp_U; ++comp) 
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            if (period.isPeriodic(idim)) 
            {
                U_bcs[comp].setLo(idim, amrex::BCType::int_dir);
                U_bcs[comp].setHi(idim, amrex::BCType::int_dir);
            } 
            else 
            {
                U_bcs[comp].setLo(idim, amrex::BCType::ext_dir);
                U_bcs[comp].setHi(idim, amrex::BCType::ext_dir);
            }
        }
    }
    
    // 4. Populate terrain boundary conditions records (foextrap or int_dir if periodic)
    for (int comp = 0; comp < ncomp_Terrain; ++comp) 
    {
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            if (period.isPeriodic(idim)) 
            {
                Terrain_bcs[comp].setLo(idim, amrex::BCType::int_dir);
                Terrain_bcs[comp].setHi(idim, amrex::BCType::int_dir);
            }
            else
            {
                Terrain_bcs[comp].setLo(idim, amrex::BCType::foextrap);
                Terrain_bcs[comp].setHi(idim, amrex::BCType::foextrap);
            }
        }
    }
}


void AmrMeshState::ResizeLevels(int nlevs) {
    U_new.resize(nlevs);
    U_old.resize(nlevs);
    DynamicTerrain.resize(nlevs);

    t_new.resize(nlevs);
    t_old.resize(nlevs);
    dt.resize(nlevs);

    flux_reg.resize(nlevs+1);
    istep.resize(nlevs, 0);
    nsubsteps.resize(nlevs, 1);
}


void AmrMeshState::Initialize() {
    
    amrex::Real time = 0.0;
    int initial_step = 0;

    if (restart_chkfile == "") {
        // start simulation from the beginning

        InitializeSolver();
        InitializeTerrainFluid();
        InitFromScratch(time); // Calls PostProcessBaseGrids to prune the mesh using NODATA
        //PostInit(); // Post Init Routine to fill lev < static_terrain_lev

        // deallocating initial static fluid MultiFab pointer since we
        // already allocated the initial DynamicFluid Multifab and now refinement or coarsening
        // is tackled by physics
        StaticFluid.reset();
    }

    IO->WritePlotfile(
        U_new, DynamicTerrain, 
        initial_step, time, 
        Geom(), refRatio(), 
        finest_level
    );
}


void AmrMeshState::InitializeTerrainFluid() {

    amrex::IntVect dom_lo(AMREX_D_DECL(0, 0, 0));
    amrex::IntVect dom_hi(AMREX_D_DECL(metadata.global_nx - 1, metadata.global_ny - 1, 0));
    amrex::Box domain(dom_lo, dom_hi);

    // 1. Define the dedicated Geometry for the Static Terrain.
    // We borrow the physical bounding box (RealBox), coordinate system, 
    // and periodicity from the base AMR level (Level 0) so they perfectly align in physical space.
    amrex::Vector<amrex::Real> prob_lo = { metadata.prob_lo_x, metadata.prob_lo_y, 0.0 };
    amrex::Vector<amrex::Real> prob_hi = { metadata.prob_hi_x, metadata.prob_hi_y, 0.0 }; 
    amrex::RealBox rb(prob_lo.data(), prob_hi.data());
    
    //amrex::RealBox rb = Geom(0).ProbDomain();
    
    int coord = Geom(0).Coord();
    amrex::Array<int, AMREX_SPACEDIM> is_per = Geom(0).isPeriodic();
    
    static_geom.define(domain, rb, coord, is_per);
    
    // 2. Distribute the newly discovered global domain evenly
    static_ba.define(domain);

    static_dm.define(static_ba);
    
    // 3. Allocate the MultiFab
    StaticTerrain.define(static_ba, static_dm, ncomp_Terrain, 2); // No ghost cells for static terrain
    
    // we make this a dynamic pointer so that we can release after initialization
    StaticFluid = std::make_unique<amrex::MultiFab>(static_ba, static_dm, ncomp_U, 2); // No ghost cells for static terrain

    StaticTerrain.setVal(-9999.0);
    StaticFluid->setVal(0.0);


    // 4. Execute the parallel hyperslab readers
    for (amrex::MFIter mfi(StaticTerrain); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& z_arr = StaticTerrain.array(mfi);
        auto const& u_arr = StaticFluid->array(mfi);
        
        IO->ReadHDF5HyperslabComponents(z_arr, bx, "terrain", 0, ncomp_Terrain, -9999.0);
        IO->ReadHDF5HyperslabComponents(u_arr, bx, "fluid", 0, ncomp_U, 0.0);
    }

    StaticTerrain.FillBoundary(static_geom.periodicity());
    StaticFluid->FillBoundary(static_geom.periodicity());
}

void AmrMeshState::InitializeSolver()
{
    if (physics_p.model == "Roe") {
        solver = Roe{};
    } else {  // fill with else ifs later
        amrex::Abort("Unknown solver type: " + physics_p.model);
    }
}