main script:
Script_learnGP -> collect data from mpc or load existing,...
design the GP or load existing, and test designed gp controller
--------------------
MyGP is a class to handle the GP design and computations; relys on the GPML toolbox

----- some functions ---------
- transform_data.m  to transform data to suitable "coordinates"
- detransform_data.m  to detransform into original coordinates
- remove_row_with_nancolumn.m  to remove a row in the dataset which...
 has a NaN in a selected column (eg to remove infeasible points)
- init_paths.m initialize paths
- get_data.m  collect MPC data randomly sampled within some defined range
- get_data_closed_loop.m collect data partially based on closed loop simulations
- data_to_GP_format.m to rearrange data in a suitable way for the gp design

--- data files ---
GP_u1_good.mat contains data for a designed GP (u1=I_st) 
GP_u2_good.mat contains data for a designed GP (u2=v_cm)
together they form one GP controller for the Fuel cell
DataFCMPC.mat contains data used to design the GPs
