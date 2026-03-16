# ComputationalMethodExam2026
Measles Epidemic Modeling, University of Padua | Computational Methods for Physics
A numerical exploration of measles epidemic dynamics, comparing Boundary Value Problem (BVP) solvers with Initial Value Problem (IVP) propagation, revealing underlying chaotic behavior.

About the Project:
This repository contains my final project for the Computational Methods for Physics course. 
The project tackles the Measles Epidemic Problem, as proposed in Širca-Horvat: Computational Methods in Physics (Section 9.9.2).
The core model is a system of ordinary differential equations (ODEs) with periodic boundary conditions. 
It describes the time evolution of three distinct categories within a population:Susceptible, Latent (Exposed), Infectious

Methodology & Code Structure
1. The "Academic" Global Approach (Code 1):
  The first script solves the problem using a global Newton-Raphson method, as suggested by the reference textbook. It treats the entire time grid as a single state vector, requiring the construction and inversion of a massive Jacobian matrix with (3(N+1))^2 elements.
Identified Limitations:
- High Computational Cost: Inverting the Jacobian matrix scales poorly with the number of time steps.
- Unnatural Initialization: It requires an initial guess for all time steps simultaneously, which is highly counter-intuitive for a dynamical system evolving through time.Convergence Issues: The method fails to converge for certain initial conditions (e.g., $y_1=0.3$ with $N \ge 1000$).

2. The Shooting Method (Codes 2 & 3)
   To overcome the limitations of the global matrix approach, the problem was recast using the Shooting Method. This approach integrates the equations forward and corrects the initial parameters using a much smaller, manageable 3*3 Damped Newton-Raphson scheme.
   Code 2 (Leap-Frog): Implements the Shooting method with a Leap-Frog integration scheme. However, it proved unstable for small values of $N$.
   Code 3 (Runge-Kutta 4): Implements the Shooting method using the RK4 integration scheme. This proved to be the most robust and efficient solver. It converges consistently and significantly faster than the global method, making it the superior choice for this specific BVP.
   3. Time Evolution and Chaos (Codes 4 - 8)
   If we assume the periodic solution is a stable attractor, treating the system as an Initial Value Problem (IVP) and propagating it forward in time should eventually converge to that periodic solution.

  Code 4 tests this hypothesis. Result: The system does not converge to the expected periodic orbit.
To understand why, Codes 5 through 8 conduct a deep dive into the system's dynamics:
  Code 5: Conducts a purely statistical investigation of the time series.
  Code 6: Generates a phase portrait to visualize the trajectories.
  Codes 7 & 8: Construct Poincaré maps (Code 8 allows for custom initial conditions to test different scenarios).
  
Conclusion
The study concludes that while the periodic boundary value problem can be successfully solved using the Shooting Method + Damped Newton-Raphson, the system's natural time evolution (IVP) does not settle into this periodic state. Instead, the Poincaré maps reveal that the IVP exhibits chaotic motion constrained within a well-defined area in the phase space.
