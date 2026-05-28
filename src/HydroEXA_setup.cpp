
#include <hydroexa/HydroEXA.H>

using namespace amrex;

int HydroEXA::num_state_data_types = NUM_STATE_DATA_TYPE;

static Box the_same_box (const Box& b) { return b; }
//static Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }

//
// Components are:
//  Interior, Inflow, Outflow,  Symmetry,     SlipWall,     NoSlipWall
//
static int scalar_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_even
};

static int norm_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_odd,  BCType::reflect_odd,  BCType::reflect_odd
};

static int tang_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_odd
};

static
void
set_scalar_bc (BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        bc.setLo(i,scalar_bc[lo_bc[i]]);
        bc.setHi(i,scalar_bc[hi_bc[i]]);
    }
}

static
void
set_x_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,norm_vel_bc[lo_bc[0]]);
    bc.setHi(0,norm_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

static
void
set_y_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,norm_vel_bc[lo_bc[1]]);
    bc.setHi(1,norm_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
#endif
}

static
void
set_z_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);
#if (AMREX_SPACEDIM >= 2)
    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);
#endif
#if (AMREX_SPACEDIM == 3)
    bc.setLo(2,norm_vel_bc[lo_bc[2]]);
    bc.setHi(2,norm_vel_bc[hi_bc[2]]);
#endif
}

void
HydroEXA::variableSetUp ()
{

    read_params();

    bool state_data_extrap = false;
    bool store_in_checkpoint = true;

    // =======================================================
    // 1. SETUP FLUID STATE (h, hu, hv)
    // =======================================================

    desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
                           StateDescriptor::Point,NUM_GROW,NUM_STATE,
                           &cell_cons_interp,state_data_extrap,store_in_checkpoint);

    Vector<BCRec>       bcs(NUM_STATE);
    Vector<std::string> name(NUM_STATE);
    BCRec bc;
    int cnt = 0;
    set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "h";
    cnt++; set_x_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "hu";
    cnt++; set_y_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "hv";

    StateDescriptor::BndryFunc bndryfunc(HydroEXA_bcfill);
    bndryfunc.setRunOnGPU(true);  // I promise the bc function will launch gpu kernels.

    desc_lst.setComponent(State_Type,
                          H,
                          name,
                          bcs,
                          bndryfunc);

    // =======================================================
    // 2. SETUP TERRAIN STATE (Bathymetry & Roughness)
    // =======================================================

    desc_lst.addDescriptor(Terrain_Type, IndexType::TheCellType(),
                           StateDescriptor::Point, NUM_GROW, NUM_TERRAIN, 
                           &cell_cons_interp, state_data_extrap, store_in_checkpoint);

    Vector<BCRec>       terrain_bcs(NUM_TERRAIN);
    Vector<std::string> terrain_names(NUM_TERRAIN);
    BCRec terrain_bc;
    
    // Often, bathymetry uses simple extrapolation (neumann) at boundaries
    set_scalar_bc(terrain_bc, phys_bc); 
    
    terrain_bcs[0] = terrain_bc; terrain_names[0] = "bathymetry";
    terrain_bcs[1] = terrain_bc; terrain_names[1] = "roughness";

    // You can use the same boundary function or write a specific one for terrain
    StateDescriptor::BndryFunc terrain_bndryfunc(HydroEXA_bcfill);
    terrain_bndryfunc.setRunOnGPU(true);

    desc_lst.setComponent(Terrain_Type, 0, terrain_names, terrain_bcs, terrain_bndryfunc);

    // =======================================================

    num_state_data_types = desc_lst.size();
}

void
HydroEXA::variableCleanUp ()
{
    desc_lst.clear();
}
