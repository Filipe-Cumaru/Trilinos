#ifndef _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DECL_HPP
#define _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DECL_HPP

#include <FROSch_GDSWInterfacePartitionOfUnity_def.hpp>

namespace FROSch {
    using namespace Teuchos;
    using namespace Xpetra;

    // Forward declaration of the solver class.
    template <class SC, class LO, class GO, class NO>
    class Solver;

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
        ConstXMatrixPtr K_;
        ConstXMatrixPtr overlappingK;
        ConstXMatrixPtr localK;

        ConstXMatrixPtr diagInteriorRowSum;
        ConstXMatrixPtr diagLeavesRowSum;

        XMapPtr repeatedMap;
        XMapPtr serialRepeatedMap;

        Array<GO> interiorDofs;
        Array<GO> interfaceDofs;
        Array<GO> rootDofs;
        Array<GO> leafDofs;

        RCP<basic_FancyOStream<char>> blackHoleStream;

    private:
        Array<GO> getEntityDofs(InterfaceEntityPtr entity) const;

        Array<GO> getEntitySetDofs(EntitySetConstPtr entitySet) const;

        void initializeOverlappingMatrices();

        void initializeMaps();

        void initializeDofsArrays();

        RCP<const Matrix<SC,LO,GO,NO>> assembleDiagSumMatrix(const Array<GO>& colIndices) const;

        RCP<Solver<SC, LO, GO, NO>> initializeLocalInterfaceSolver(const XMatrixPtr kII,
                                                                   const XMatrixPtr diagSumInterior,
                                                                   const XMatrixPtr diagSumExtra = null) const;
        
        void addAncestorTerm(const InterfaceEntityPtr entity,
                             Array<GO> entityDofs,
                             Array<GO> entityRootsDofs,
                             const SolverPtr kBBSolver,
                             XMultiVectorPtr mVPhiBV) const;
    };
}

#endif
