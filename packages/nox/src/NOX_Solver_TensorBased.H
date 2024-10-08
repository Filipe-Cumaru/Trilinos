// @HEADER
// *****************************************************************************
//            NOX: An Object-Oriented Nonlinear Solver Package
//
// Copyright 2002 NTESS and the NOX contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef NOX_SOLVER_TENSORBASED_H
#define NOX_SOLVER_TENSORBASED_H

#include "NOX_Solver_Generic.H"        // base class
#include "Teuchos_ParameterList.hpp"        // class data element
#include "NOX_Utils.H"                // class data element
#include "Teuchos_RCP.hpp"  // class data element

#include "NOX_LineSearch_Generic.H" // base class
#include "NOX_LineSearch_Utils_Printing.H"  // class data member
#include "NOX_LineSearch_Utils_Slope.H"     // class data member


namespace NOX {

  namespace Direction {
    class Generic;
  }

  class LineSearchCounters;
  class Observer;

namespace Solver {

/*!
  \brief Nonlinear solver based on a rank-1 tensor method.

  Solves \f$ F(x)=0\f$ using a rank-1 tensor method and a linesearch
  globalization.

  At the kth nonlinear iteration, the solver does the following:

  <ul>

  <li>Computes the tensor direction \f$ d_T \f$ by finding the root or
      smallest magnitude minimizer of the local model \f[ M_T(x_k+d) =
      F_k + J_kd + a_k(s_k^Td)^2, \f] where \f[ a_k = 2(F_{k-1} - F_k
      - J_ks_k) / (s_k^Ts_k)^2 \f] and \f[ s_k = s_{k-1} - s_k. \f]

  <li>Modifies the step according to a global strategy and updates
      \f$ x\f$ as \f$ x_{k+1} = x_k + d(\lambda) \f$ via a linesearch
      method, where \f$ d(\lambda) \f$ is some function of \f$ \lambda
      \f$.  For instance, the curvilinear step \f$ d_{\lambda T} \f$
      is a function of the linesearch parameter \f$ \lambda \f$ and is
      a parametric step that spans the directions of the tensor step
      and the Newton step.  At \f$ \lambda=1 \f$, the curvilinear step
      equals the full tensor step, and as \f$ \lambda \f$ nears 0, the
      curvilinear step approaches the Newton direction.  This step
      provides a monotonic decrease in the norm of the local tensor
      model as \f$ \lambda \f$ varies from 0 to 1.

  </ul>

  The solver iterates until the status tests (see NOX::StatusTest)
  determine either failure or convergence.

  <B>Input Parameters</B>

  To use this solver, set the "Nonlinear Solver" parameter to be
  "Tensor Based".  Then, specify the following sublists with the
  appropriate parameters as indicated below.

  <ul>

  <li> "Direction" - Sublist of the direction parameters, passed to
       the NOX::Direction::Factory constructor. Defaults to an empty
       list.

       <ul>

       <li> "Method" - Name of the direction to be computed in this
       solver.  "Tensor" and "Newton" are the only two valid choices.
       A sublist by this name specifies all of the parameters to be
       passed to the linear solver.  See below under "Linear Solver".

       <li> "Rescue Bad Newton Solve" (Boolean) - If the linear solve
       does not meet the tolerance specified by the forcing term, then
       use the step anyway. Defaults to true.

       <li> "Linear Solver" - Sublist for the specific linear solver
       parameters that are passed to
       NOX::Abstract::Group::computeNewton() and
       NOX::Abstract::Group::applyJacobianInverse().  "Linear Solver"
       is itself a sublist of the list specified in "Method" above
       (i.e., "Tensor" or "Newton").  Below is a partial list of
       standard parameters usually available in common linear solvers.
       Check with the specific linear solver being used for other
       parameters.

        <ul>

        <li> "Max Iterations" - Maximum number of Arnoldi
            iterations (also max Krylov space dimension)

            <li> "Tolerance" - Relative tolerance for solving local
            model [default = 1e-4]

        <li> "Output Frequency" - Print output at every number of
            iterations [default = 20]

        </ul>

       </ul>

  <li> "Line Search" - Sublist of the line search parameters.  Because
       the tensor step is not guaranteed to be a descent direction on
       the function, not all "basic" line search approaches would be
       appropriate.  Thus, the LineSearch classes available to
       Newton's method (e.g., Polynomial, More-Thuente) are not used
       here.  Instead, this solver class approriately handles
       technical considerations for tensor methods with its own set of
       global strategies.  The following parameters specify the
       specific options for this line search:

       <ul>

       <li> "Method" - Name of the line search available to tensor
       methods Valid choices are:

            <ul>

            <li> "Curvilinear" - Backtrack along the "curvilinear"
        path that spans the tensor direction and the Newton
        direction and that maintains monotonicity on the tensor
        model.  Recommended because it tends to be more robust and
        efficient than the other choices.  [Default]

        <li> "Standard" - Backtrack along tensor direction unless
        it is not a descent direction, in which case backtrack
        along Newton direction.

        <li> "Dual" - Backtrack along both the Newton and tensor
        directions and choose the better of the two.

        <li> "Full Step" - Only use the full step and do not
        backtrack along both the Newton and tensor directions and
        choose the better of the two.

            </ul>

       <li> "Lambda selection" - Flag for how to calculate the next
       linesearch parameter lambda.  Valid choices are "Quadratic" and
       "Halving" (default).  Quadratic constructs a quadratic
       interpolating polynomial from the last trial point and uses the
       minimum of this function as the next trial lambda (bounded by
       0.1).  Halving divides the linesearch parameter by 2 before
       each trial, which is simpler but tends to generate longer steps
       than quadratic.

       <li> "Default Step" - Starting value of the linesearch parameter
       (defaults to 1.0)

       <li> "Minimum Step" - Minimum acceptable linesearch parameter
       before the linesearch terminates (defaults to 1.0e-12). If
       there are many linesearch failures, then lowering this value is
       one thing to try.

       <li> "Recovery Step Type" - Determines the step size to take when the
       line search fails.  Choices are:

          <ul>

      <li> "Constant" [default] - Uses a constant value set in
      "Recovery Step".

      <li> "Last Computed Step" - Uses the last value computed by the
      line search algorithm.

      </ul>

       <li> "Recovery Step" - Step parameter to take when the line
       search fails (defaults to value for "Default Step")

       <li> "Max Iters" - Maximum number of iterations (i.e., backtracks)

       </ul>

  <li>"Solver Options" - Sublist of general solver options.

       <ul>

       <li> "User Defined Pre/Post Operator" is supported.  See
       NOX::Parameter::PrePostOperator for more details.

       </ul>

  </ul>

  <B>Output Parameters</B>

  Every time solve() is called, a sublist for output parameters called
  "Output" will be created and will contain the following parameters:

  <ul>

  <li> "Nonlinear Iterations" - Number of nonlinear iterations

  <li> "2-Norm of Residual" - L-2 norm of the final residual \f$ F(x_k) \f$.

  </ul>


<b>References</b>

  <ul>

  <li> B. W. Bader, <em>Tensor-Krylov methods for solving large-scale
  systems of nonlinear equations</em>, Ph.D. Thesis, 2003, University
  of Colorado, Boulder, Colorado.

  <li> B. W. Bader, <em>Tensor-Krylov methods for solving large-scale
  systems of nonlinear equations</em>, submitted to SIAM J. Numer. Anal.

  <li> B. W. Bader and R. B. Schnabel, <em>Curvilinear linesearch for
  tensor methods</em>, SISC, 25(2):604-622.

  <li> R. B. Schnabel and P. D. Frank, <em>Tensor methods for
  nonlinear equations</em>, SIAM J. Numer. Anal., 21(5):815-843.

  </ul>


  \author Brett Bader (SNL 9233)

*/

class TensorBased : public Generic {

public:

  /*!
    \brief Constructor

    See reset() for description.
  */
  TensorBased(const Teuchos::RCP<NOX::Abstract::Group>& grp,
          const Teuchos::RCP<NOX::StatusTest::Generic>& tests,
          const Teuchos::RCP<Teuchos::ParameterList>& params);

  //! Destructor
  virtual ~TensorBased();

  virtual void reset(const NOX::Abstract::Vector& initialGuess,
             const Teuchos::RCP<NOX::StatusTest::Generic>& tests);
  virtual void reset(const NOX::Abstract::Vector& initialGuess);
  virtual void reset();
  virtual NOX::StatusTest::StatusType step();
  virtual NOX::StatusTest::StatusType solve();
  virtual const NOX::Abstract::Group& getSolutionGroup() const;
  virtual const NOX::Abstract::Group& getPreviousSolutionGroup() const;
  virtual NOX::StatusTest::StatusType getStatus() const;
  virtual int getNumIterations() const;
  virtual const Teuchos::ParameterList& getList() const;

  inline virtual Teuchos::RCP< const NOX::Abstract::Group > getSolutionGroupPtr() const {return solnPtr;};
  inline virtual Teuchos::RCP< const NOX::Abstract::Group > getPreviousSolutionGroupPtr() const {return oldSolnPtr;};
  inline virtual Teuchos::RCP< const Teuchos::ParameterList > getListPtr() const {return paramsPtr;};
  virtual Teuchos::RCP<const NOX::SolverStats> getSolverStatistics() const;

protected:

  //! Print out initialization information.
  virtual void init();

  //! Prints the current iteration information.
  virtual void printUpdate();

  //! Constructor initialization routine.
  virtual bool reset(const Teuchos::RCP<NOX::Abstract::Group>& grp,
             const Teuchos::RCP<NOX::StatusTest::Generic>& tests,
             const Teuchos::RCP<Teuchos::ParameterList>& params);
protected:

  //! Subroutine for computing the tensor and Newton directions
  bool computeTensorDirection(NOX::Abstract::Group& soln,
                  const NOX::Solver::Generic& solver);

  //! Subroutine for calculating beta
  double calculateBeta(double qa, double qb, double qc, double& qval,
               double& lambdaBar, double lambda=1.0) const;

  //! Subroutine for computing the curvilinear step
  bool computeCurvilinearStep(NOX::Abstract::Vector& dir,
                  const NOX::Abstract::Group& soln,
                  const NOX::Solver::Generic& s,
                  double& lambda);

  //! Subroutine for executing the tensor linesearch
  bool implementGlobalStrategy(NOX::Abstract::Group& newGrp,
                   double& step,
                   const NOX::Solver::Generic& s);


  //! Performs a standard tensor linesearch (tensor or Newton direction)
  bool performLinesearch(NOX::Abstract::Group& newsoln,
             double& step,
             const NOX::Abstract::Vector& lsDir,
             const NOX::Solver::Generic& s);

  //! Compute the residual norm of the local model
  double getNormModelResidual(const NOX::Abstract::Vector& dir,
                  const NOX::Abstract::Group& soln,
                  bool isTensorModel) const;

  //! Print pertinent information about the direction
  void printDirectionInfo(std::string dirName,
              const NOX::Abstract::Vector& dir,
              const NOX::Abstract::Group& soln,
              bool isTensorModel) const;

  //! Calculate the directional derivative
  double getDirectionalDerivative(const NOX::Abstract::Vector& dir,
                  const NOX::Abstract::Group& soln) const;

  //! Select lambda for linesearch (quadratic or halving)
  double selectLambda(double newf, double oldf,
              double oldfprime, double lambda);

  //! Throw an error with a method's name and error message
  void throwError(const std::string& functionName,
          const std::string& errorMsg) const;

protected:

  //! Pointer to the global data object.
  Teuchos::RCP<NOX::GlobalData> globalDataPtr;

  //! Printing Utils
  Teuchos::RCP<NOX::Utils> utilsPtr;

  //! Current solution.
  Teuchos::RCP<NOX::Abstract::Group> solnPtr;

  //! Previous solution pointer.
  /*! We have both a pointer and a reference because we need to create
    a DERIVED object and then want to have a reference to it. */
  Teuchos::RCP<NOX::Abstract::Group> oldSolnPtr;

  //! Current Newton direction (pointer).
  /*! We have both a pointer and a reference because we need to create
    a DERIVED object and then want to have a reference to it. */
  Teuchos::RCP<NOX::Abstract::Vector> newtonVecPtr;

  //! Current tensor direction (pointer).
  /*! We have both a pointer and a reference because we need to create
    a DERIVED object and then want to have a reference to it. */
  Teuchos::RCP<NOX::Abstract::Vector> tensorVecPtr;

  //! Current tensor term vector (pointer).
  /*! We have both a pointer and a reference because we need to create
    a DERIVED object and then want to have a reference to it. */
  Teuchos::RCP<NOX::Abstract::Vector> aVecPtr;

  //! Vector to previous point (pointer).
  /*! We have both a pointer and a reference because we need to create
    a DERIVED object and then want to have a reference to it. */
  Teuchos::RCP<NOX::Abstract::Vector> sVecPtr;

  //! Working vector (pointer).
  /*! We have both a pointer and a reference because we need to create
    a DERIVED object and then want to have a reference to it. */
  Teuchos::RCP<NOX::Abstract::Vector> tmpVecPtr;

  //! Residual vector (pointer).
  /*! We have both a pointer and a reference because we need to create
    a DERIVED object and then want to have a reference to it. */
  Teuchos::RCP<NOX::Abstract::Vector> residualVecPtr;

  //! Stopping test.
  Teuchos::RCP<NOX::StatusTest::Generic> testPtr;

  //! Type of check to use for status tests.  See NOX::StatusTest for more details.
  NOX::StatusTest::CheckType checkType;

  //! Input parameters.
  Teuchos::RCP<Teuchos::ParameterList> paramsPtr;

  //! Line Search parameters.
  //Teuchos::ParameterList& lsParams;

  //! Direction parameters.
  //Teuchos::ParameterList& dirParams;

  //! Parameters for the Linear Solver of the local model.
  Teuchos::ParameterList* linearParamsPtr;

  //! Current step.
  double stepSize;

  //! Value of sc'*dt.
  double beta;

  //! Number of nonlinear iterations.
  int nIter;

  //! %Status of nonlinear solver.
  NOX::StatusTest::StatusType status;

  //! Types of steps
  enum StepType
    {
      TensorStep,
      NewtonStep
    };

  //! Flag for the base direction to compute after the first iteration
  StepType requestedBaseStep;

  //! Flag for the direction to be computed this iteration
  //stepType requestedStep;

  //! Enumerated list for each type of line search
  enum LineSearchType
    {
      Curvilinear,
      Standard,
      Dual,
      FullStep,
      Newton
    };

  //! Choice of line search
  LineSearchType lsType;

  //! Algorithms used to determine convergence of the line search
  enum ConvergenceCriteriaType
  {
    //! Sufficient decrease condition
    ArmijoGoldstein,
    //! Ared/Pred condition
    AredPred,
    //! Just accept the first step
    None
  };

  //! Choice of convergence criteria   (currently unused)
  ConvergenceCriteriaType convCriteria;

  //! Types of lambda selection
  enum LambdaSelectionType
  {
    Halving,
    Quadratic
  };

  //! Flag for lambda selection (Halving/Quadratic)
  LambdaSelectionType lambdaSelection;

  //! Type of recovery step to use
  enum RecoveryStepType {
    //! Use a constant value
    Constant,
    //! Use the last value computed in the line search algorithm
    LastComputedStep
  };

  //! Choice of the recovery step type; uses "Recovery Step Type" parameter
  RecoveryStepType recoveryStepType;

  //! Flag for using modifications that force quadratic to have real root
  bool useModifiedMethod;

  //! Flag for Newton direction
  bool isNewtonDirection;

  //! Flag for rescuing Linear Solver from a bad solve
  bool doRescue;

  //! Minimum step length (i.e., when we give up)
  double minStep;

  //! Default step
  double defaultStep;

  //! Default step for linesearch failure
  double recoveryStep;

  //! Maximum iterations
  int maxIters;

  //! Scaling factor for the Armijo-Goldstein condition
  double alpha;

  //! Value of s'*inv(J)*F
  double sTinvJF;

  //! Value of s'*inv(J)*a
  double sTinvJa;

  //! Common line search printing utilities.
  Teuchos::RCP<NOX::LineSearch::Utils::Printing> print;

  //! Common common counters for line searches.
  NOX::LineSearchCounters* counter;

  //! Common slope calculations for line searches.
  NOX::LineSearch::Utils::Slope slopeObj;

  //! Counter for number of Jacobian-vector products
  mutable int numJvMults;

  //! Counter for number of "double" Jacobian-vector products
  mutable int numJ2vMults;

  //! Pointer to a user defined NOX::Observer object.
  Teuchos::RCP<NOX::Observer> observer;

};
} // namespace Solver
} // namespace NOX

#endif
