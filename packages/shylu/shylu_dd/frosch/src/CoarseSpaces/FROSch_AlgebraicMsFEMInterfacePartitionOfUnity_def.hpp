#ifndef _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DEF_HPP
#define _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DEF_HPP

#include <FROSch_AlgebraicMsFEMInterfacePartitionOfUnity_decl.hpp>

namespace FROSch {
    template <class SC, class LO, class GO, class NO>
    AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::AlgebraicMsFEMInterfacePartitionOfUnity(CommPtr mpiComm,
                                                                                                     CommPtr serialComm,
                                                                                                     UN dimension,
                                                                                                     UN dofsPerNode,
                                                                                                     ConstXMapPtr nodesMap,
                                                                                                     ConstXMapPtrVecPtr dofsMaps,
                                                                                                     ParameterListPtr parameterList,
                                                                                                     ConstXMatrixPtr K,
                                                                                                     Verbosity verbosity,
                                                                                                     UN levelID) : GDSWInterfacePartitionOfUnity<SC, LO, GO, NO>(mpiComm,
                                                                                                                                                                 serialComm,
                                                                                                                                                                 dimension,
                                                                                                                                                                 dofsPerNode,
                                                                                                                                                                 nodesMap,
                                                                                                                                                                 dofsMaps,
                                                                                                                                                                 parameterList,
                                                                                                                                                                 verbosity,
                                                                                                                                                                 levelID),
                                                                                                                   K_(K) {
        this->UseVertices_ = false;
        this->UseShortEdges_ = false;
        this->UseStraightEdges_ = false;
        this->UseEdges_ = false;
        this->UseFaces_ = false;
        this->LocalPartitionOfUnity_ = ConstXMultiVectorPtrVecPtr(1);
        this->PartitionOfUnityMaps_ = ConstXMapPtrVecPtr(1);
    }

    template <class SC, class LO, class GO, class NO>
    Array<GO> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::getEntityDofs(InterfaceEntityPtr entity) {
        UN dofsPerNode = this->DDInterface_->getInterface()->getEntity(0)->getDofsPerNode();
        Array<GO> entityDofs;
        for (UN i = 0; i < entity->getNumNodes(); i++) {
            for (UN j = 0; j < dofsPerNode; j++) {
                entityDofs.append(entity->getGlobalDofID(i, j));
            }
        }
        return entityDofs;
    }

    template <class SC, class LO, class GO, class NO>
    Array<GO> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::getEntitySetDofs(EntitySetConstPtr entitySet) {
        Array<GO> entitySetDofs;
        for (UN i = 0; i < entitySet->getNumEntities(); i++) {
            InterfaceEntityPtr tmpEntity = entitySet->getEntity(i);
            Array<GO> entityDofs = this->getEntityDofs(tmpEntity);
            for (UN j = 0; j < entityDofs.size(); j++) {
                entitySetDofs.append(entityDofs[j]);
            }
        }
        return entitySetDofs;
    }

    template <class SC, class LO, class GO, class NO>
    int AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::computePartitionOfUnity(ConstXMultiVectorPtr nodeList)
    {
        return GDSWInterfacePartitionOfUnity<SC, LO, GO, NO>::computePartitionOfUnity(nodeList);
    }
}

#endif
