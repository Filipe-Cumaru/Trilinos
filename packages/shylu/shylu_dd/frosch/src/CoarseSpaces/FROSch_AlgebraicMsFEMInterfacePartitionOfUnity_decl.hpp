#ifndef _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DECL_HPP
#define _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DECL_HPP

#include <FROSch_GDSWInterfacePartitionOfUnity_def.hpp>

namespace FROSch
{
    template <class SC = double,
              class LO = int,
              class GO = DefaultGlobalOrdinal,
              class NO = Tpetra::KokkosClassic::DefaultNode::DefaultNodeType>
    class AlgebraicMsFEMInterfacePartitionOfUnity : public GDSWInterfacePartitionOfUnity<SC, LO, GO, NO>
    {
    protected:
        using UN = typename PartitionOfUnity<SC, LO, GO, NO>::UN;

        using CommPtr = typename PartitionOfUnity<SC, LO, GO, NO>::CommPtr;

        using XMapPtr = typename PartitionOfUnity<SC, LO, GO, NO>::XMapPtr;
        using ConstXMapPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMapPtr;
        using ConstXMapPtrVecPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMapPtrVecPtr;

        using ParameterListPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ParameterListPtr;

        using XMultiVector = typename PartitionOfUnity<SC, LO, GO, NO>::XMultiVector;
        using ConstXMultiVectorPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMultiVectorPtr;
        using ConstXMultiVectorPtrVecPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMultiVectorPtrVecPtr;

        using XMatrixPtr = typename PartitionOfUnity<SC, LO, GO, NO>::XMatrixPtr;
        using ConstXMatrixPtr = typename PartitionOfUnity<SC, LO, GO, NO>::ConstXMatrixPtr;

        using EntitySetPtr = typename PartitionOfUnity<SC, LO, GO, NO>::EntitySetPtr;
        using EntitySetConstPtr = const EntitySetPtr;

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
    };
}

#endif
