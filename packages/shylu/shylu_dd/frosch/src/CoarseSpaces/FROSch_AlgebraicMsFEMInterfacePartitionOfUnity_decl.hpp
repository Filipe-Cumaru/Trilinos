#ifndef _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DECL_HPP
#define _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DECL_HPP

#include <FROSch_GDSWInterfacePartitionOfUnity_def.hpp>

namespace FROSch {
    using namespace Teuchos;
    using namespace Xpetra;

    // Forward declaration of the solver class.
    template <class SC, class LO, class GO, class NO>
    class Solver;

    /**
     * Implementation of an interface partition of unity function based on the
     * algebraic formulation of the multiscale finite element method (MsFEM).
     * For more details, see https://doi.org/10.48550/arXiv.2408.08187.
     */
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class AlgebraicMsFEMInterfacePartitionOfUnity : public GDSWInterfacePartitionOfUnity<SC, LO, GO, NO> {
    protected:
        using UN = typename PartitionOfUnity<SC, LO, GO, NO>::UN;

        using CommPtr = typename PartitionOfUnity<SC, LO, GO, NO>::CommPtr;

        using XMapPtr = typename PartitionOfUnity<SC, LO, GO, NO>::XMapPtr;
        using ConstXMapPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMapPtr;
        using ConstXMapPtrVecPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMapPtrVecPtr;

        using ParameterListPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ParameterListPtr;

        using XMultiVector = typename PartitionOfUnity<SC, LO, GO, NO>::XMultiVector;
        using XMultiVectorPtr = typename PartitionOfUnity<SC, LO, GO, NO>::XMultiVectorPtr;
        using ConstXMultiVectorPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMultiVectorPtr;
        using ConstXMultiVectorPtrVecPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMultiVectorPtrVecPtr;

        using XMatrix = typename PartitionOfUnity<SC, LO, GO, NO>::XMatrix;
        using XMatrixPtr = typename PartitionOfUnity<SC, LO, GO, NO>::XMatrixPtr;
        using ConstXMatrixPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMatrixPtr;

        using EntitySetPtr = typename PartitionOfUnity<SC, LO, GO, NO>::EntitySetPtr;
        using EntitySetConstPtr = const EntitySetPtr;
        using EntitySetPtrVecPtr = typename PartitionOfUnity<SC, LO, GO, NO>::EntitySetPtrVecPtr;

        using InterfaceEntityPtr = typename PartitionOfUnity<SC, LO, GO, NO>::InterfaceEntityPtr;

        using SolverPtr = RCP<Solver<SC, LO, GO, NO>>;

    public:
        AlgebraicMsFEMInterfacePartitionOfUnity(CommPtr mpiComm,
                                                CommPtr serialComm,
                                                UN dimension,
                                                UN dofsPerNode,
                                                ConstXMapPtr nodesMap,
                                                ConstXMapPtrVecPtr dofsMaps,
                                                ParameterListPtr parameterList,
                                                ConstXMatrixPtr K,
                                                Verbosity verbosity = All,
                                                UN levelID = 1);

        virtual int computePartitionOfUnity(ConstXMultiVectorPtr nodeList = null);

    protected:
        // The system matrix.
        ConstXMatrixPtr K_;

        // The system matrix defined on a repeated map.
        ConstXMatrixPtr overlappingK;

        // The matrix K_ defined on a serial repeated map.
        ConstXMatrixPtr localK;

        // Diagonal matrix containing the global sum of the rows corresponding
        // to the interior dofs, i.e. diagInteriorRowSum = diag(K * R_I^T * 1_I),
        // where R_I is a restriction matrix to the interior and 1_I is a vector
        // of ones.
        ConstXMatrixPtr diagInteriorRowSum;

        // Analogous to diagInteriorRowSum for the leaf dofs.
        ConstXMatrixPtr diagLeavesRowSum;

        // A non-unique map containing all the dofs in the unique map plus
        // the interface dofs.
        XMapPtr repeatedMap;

        // The repeated map defined on a serial communicator.
        XMapPtr serialRepeatedMap;

        // Teuchos array containing all interior dofs.
        Array<GO> interiorDofs;

        // Teuchos array containing all interface dofs.
        Array<GO> interfaceDofs;

        // Teuchos array containing all root dofs.
        Array<GO> rootDofs;

        // Teuchos array containing all leaf dofs.
        Array<GO> leafDofs;

        RCP<basic_FancyOStream<char>> blackHoleStream;

    private:
        /**
         * \brief Retrieves the global indices of the dofs in the input interface entity.
         * 
         * \param[in ] entity The interface entity for which the indices will be queried.
         * 
         * \return A Teuchos array containing the global indices of the dofs in `entity`.
         */
        Array<GO> getEntityDofs(InterfaceEntityPtr entity) const;

        /**
         * \brief Retrieves the global indices of the dofs in the entity set.
         * 
         * \param[in ] entitySet The entity set for which the indices will be queried.
         * 
         * \return A Teuchos array containing the global indices of the dofs in `entitySet`.
         */
        Array<GO> getEntitySetDofs(EntitySetConstPtr entitySet) const;

        /**
         * \brief Initializes the matrices `localK` and `overlappingK`.
         */
        void initializeOverlappingMatrices();

        /**
         * \brief Initializes the maps `repeatedMap` and `serialRepeatedMap`.
         */
        void initializeMaps();

        /**
         * \brief Initializes the arrays `interiorDofs`, `interfaceDofs` and `leafDofs`.
         */
        void initializeDofsArrays();

        /**
         * \brief Computes a diagonal matrix containing the sum of each row over
         * the columns in `colIndices`.
         * 
         * This member function performs the operation
         * D = diag(overlappingK * R_C^T 1_C)
         * where C corresponds to the set of global indices in `colIndices`,
         * R_C is a restriction matrix into C, 1_C is a vector of dimension |C|
         * containing only ones, and diag converts the resulting vector into a
         * diagonal matrix. This is equivalent to summing the entries in each row
         * that correspond to the columns in `colIndices`.
         * 
         * \param[in ] colIndices The global indices of the columns that should 
         * be taken into account in the sum.
         */
        RCP<const Matrix<SC,LO,GO,NO>> assembleDiagSumMatrix(const Array<GO>& colIndices) const;

        /**
         * \brief Initializes a pointer to a solver object for a modified version
         * of `kII`.
         * 
         * This member function carries out two operations:
         * 1. Perform the elimination procedure from the algebraic MsFEM formulation; and
         * 2. Initialize a solver object using the modified version of kII.
         * The first step amounts to computing:
         * `kIIMod = kII + diagSumInterior + diagSumExtra`.
         * If `diagSumExtra` is `null`, then it computes:
         * `kIIMod = kII + diagSumInterior`.
         * After this, the solver objected is initialized using `kIIMod`.
         * 
         * \param[in ] kII The matrix used in the elimination procedure and for
         * which the solver is computed
         * 
         * \param[in ] diagSumInterior A diagonal matrix corresponding to the sum
         * of the entries related to the interior nodes.
         * 
         * \param[in ] diagSumExtra A diagonal matrix corresponding to the sum
         * of the entries related to other interface nodes (if needed). Defaults to null.
         */
        RCP<Solver<SC, LO, GO, NO>> initializeLocalInterfaceSolver(const XMatrixPtr kII,
                                                                   const XMatrixPtr diagSumInterior,
                                                                   const XMatrixPtr diagSumExtra = null) const;
        
        /**
         * \brief Computes an additional term for the IPOU related to any ancestor of `entity`.
         * 
         * For interface entities that are leaves, the formulation of the
         * algebraic MsFEM IPOU involves an additional term besides the
         * extension from the root dofs.
         * In global notation:
         * \Phi_{L} = -\tilde{K}_{LL}^{-1} * K_{LV} + \tilde{K}_{LL}^{-1} * K_{LR} * \tilde{K}_{LL}^{-1} * K_{LV}
         *            |____________(I)_____________|  |_________________________(II)____________________________|
         * where the indices V, L and R correspond to root, leaf and the remaining
         * interface entities. This member function computes the term (II) assuming
         * that `mVPhiBV` already contains (I).
         * 
         * \param[in ] entity The interface entity for which the term will be computed.
         * 
         * \param[in ] entityDofs The global indices of the dofs in `entity`.
         * 
         * \param[in ] entityRootsDofs The global indices of the dofs of the
         * roots of `entity`.
         * 
         * \param[in ] kBBSolver A solver object for the modified matrix block
         * related to the dofs in `entity`.
         * 
         * \param[inout ] mVPhiBV The IPOU values related to the dofs in `entity`.
         * The correction term will be added to this MultiVector.
         */
        void addAncestorTerm(const InterfaceEntityPtr entity,
                             Array<GO> entityDofs,
                             Array<GO> entityRootsDofs,
                             const SolverPtr kBBSolver,
                             XMultiVectorPtr mVPhiBV) const;
    };
}

#endif
