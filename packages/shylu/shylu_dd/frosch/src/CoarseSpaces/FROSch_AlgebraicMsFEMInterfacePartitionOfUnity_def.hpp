#ifndef _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DEF_HPP
#define _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DEF_HPP

#include <FROSch_AlgebraicMsFEMInterfacePartitionOfUnity_decl.hpp>
#include <FROSch_SolverFactory_def.hpp>
#include <Xpetra_MatrixMatrix_def.hpp>

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
        this->assembleLocalMatrix();
    }

    template <class SC, class LO, class GO, class NO>
    int AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::computePartitionOfUnity(ConstXMultiVectorPtr nodeList) {
        FROSCH_DETAILTIMER_START_LEVELID(computePartitionOfUnityTime,
                                         "AlgebraicMsFEMInterfacePartitionOfUnity::computePartitionOfUnity");

        UN dofsPerNode = this->DDInterface_->getInterface()->getEntity(0)->getDofsPerNode();
        UN numInterfaceDofs = dofsPerNode * this->DDInterface_->getInterface()->getEntity(0)->getNumNodes();

        // Initialization of the interface data structures.
        this->DDInterface_->buildEntityHierarchy();
        this->DDInterface_->buildEntityMaps(false,  // vertices
                                            false,  // short edges
                                            false,  // straight edges
                                            false,  // edges
                                            false,  // faces
                                            true,   // roots
                                            false); // leaves

        EntitySetPtrVecPtr entitySetVector = this->DDInterface_->getEntitySetVector();

        // Retrieve all root entities owned by the process.
        EntitySetPtr allRoots = this->DDInterface_->getRoots();
        this->PartitionOfUnityMaps_[0] = allRoots->getEntityMap();

        // Interior entities and their dofs.
        EntitySetConstPtr interiorSet = this->DDInterface_->getInterior();
        Array<GO> interiorDofs = this->getEntitySetDofs(interiorSet);

        // Initialization of a vector to store the IPOU values.
        XMapPtr serialInterfaceMap = MapFactory<LO, GO, NO>::Build(this->DDInterface_->getNodesMap()->lib(),
                                                                   numInterfaceDofs,
                                                                   0,
                                                                   this->SerialComm_);
        XMultiVectorPtr ipouVector = MultiVectorFactory<SC, LO, GO, NO>::Build(serialInterfaceMap,
                                                                               allRoots->getNumEntities());

        RCP<basic_FancyOStream<char>> blackHoleStream = getFancyOStream(rcp(new oblackholestream()));
        for (UN i = 0; i < entitySetVector.size(); i++) {
            for (UN j = 0; j < entitySetVector[i]->getNumEntities(); j++) {
                InterfaceEntityPtr currEntity = entitySetVector[i]->getEntity(j);
                LO rootId = currEntity->getRootID();

                if (rootId == -1) {
                    // If not a coarse node, compute the interface IPOU
                    // value using the algebraic MsFEM/AMS approach for
                    // edge/face nodes.
                    UN numRoots = currEntity->getRoots()->getNumEntities();
                    FROSCH_ASSERT(numRoots != 0, "rootID==-1 but numRoots==0!");

                    EntitySetPtr roots = currEntity->getRoots();
                    Array<GO> rootsDofs = this->getEntitySetDofs(roots);
                    Array<GO> currEntityDofs = this->getEntityDofs(currEntity);

                    // Exract the submatrices required to assemble the IPOU.
                    // These are equivalent to the blocks in the wirebasket order:
                    //           (kII kIB kIV)
                    // W^T K W = (kBI kBB kBV)
                    //           (kVI kVB kVV)
                    XMatrixPtr kBI;
                    XMatrixPtr kBB;
                    XMatrixPtr kBV;
                    BuildSubmatrix(this->localK, currEntityDofs(), kBB);
                    BuildSubmatrix(this->localK,
                                   currEntityDofs(),
                                   rootsDofs(),
                                   kBV);
                    BuildSubmatrix(this->localK,
                                   currEntityDofs(),
                                   interiorDofs(),
                                   kBI);

                    // kBIRowSum = kBI * 1_I, 1_I = (1 ... 1)
                    // This is equivalent to adding the rows of kBI.
                    XMultiVectorPtr kBIRowSum = MultiVectorFactory<SC, LO, GO, NO>::Build(kBI->getRangeMap(), 1);
                    XMultiVectorPtr ones = MultiVectorFactory<SC, LO, GO, NO>::Build(kBI->getDomainMap(), 1);
                    ones->putScalar(ScalarTraits<SC>::one());
                    kBI->apply(*ones, *kBIRowSum);

                    // kBBMod = kBB + diag(kBIRowSum)
                    XMatrixPtr diagKBI = MatrixFactory<SC, LO, GO, NO>::Build(kBIRowSum->getVector(0));
                    XMatrixPtr kBBMod;
                    MatrixMatrix<SC, LO, GO, NO>::TwoMatrixAdd(*kBB,
                                                               false,
                                                               ScalarTraits<SC>::one(),
                                                               *diagKBI,
                                                               false,
                                                               ScalarTraits<SC>::one(),
                                                               kBBMod,
                                                               *blackHoleStream,
                                                               false);
                    kBBMod->fillComplete();

                    // Initialization of the interface solver.
                    SolverPtr kBBSolver = SolverFactory<SC, LO, GO, NO>::Build(kBBMod,
                                                                               sublist(this->ParameterList_, "InterfaceSolver"),
                                                                               string(""));
                    kBBSolver->initialize();
                    kBBSolver->compute();

                    // Convert kBV to a MultiVector so it can be used in
                    // kBBSolver->apply(...).
                    XMultiVectorPtr mVkBV = MultiVectorFactory<SC, LO, GO, NO>::Build(kBB->getDomainMap(),
                                                                                      rootsDofs.size());
                    ArrayView<const LO> localColIdx;
                    ArrayView<const SC> localVals;
                    for (UN k = 0; k < kBV->getLocalNumRows(); k++) {
                        kBV->getLocalRowView(k, localColIdx, localVals);
                        for (UN l = 0; l < localColIdx.size(); l++) {
                            mVkBV->replaceLocalValue(k, localColIdx[l], localVals[l]);
                        }
                    }

                    // Compute the solution on the interface.
                    XMultiVectorPtr mVPhiBV = MultiVectorFactory<SC, LO, GO, NO>::Build(kBB->getDomainMap(),
                                                                                        rootsDofs.size());
                    kBBSolver->apply(*mVkBV, *mVPhiBV);

                    // Set the entries on the IPOU vector.
                    for (UN k = 0; k < numRoots; k++) {
                        LO rootIdx = roots->getEntity(k)->getRootID();
                        ArrayRCP<const SC> mVPhiBVk = mVPhiBV->getData(k);
                        UN n = 0;
                        for (UN l = 0; l < currEntity->getNumNodes(); l++) {
                            for (UN m = 0; m < dofsPerNode; m++) {
                                ipouVector->replaceLocalValue(currEntity->getGammaDofID(l, m),
                                                              rootIdx,
                                                              -mVPhiBVk[n]);
                                n += 1;
                            }
                        }
                    }
                } else {
                    // If coarse node, fill in the IPOU function with ones.
                    for (UN k = 0; k < currEntity->getNumNodes(); k++) {
                        for (UN l = 0; l < dofsPerNode; l++) {
                            ipouVector->replaceLocalValue(currEntity->getGammaDofID(k, l),
                                                          rootId,
                                                          ScalarTraits<SC>::one());
                        }
                    }
                }
            }
        }

        this->LocalPartitionOfUnity_[0] = ipouVector;

        return 0;
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
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::assembleLocalMatrix() {
        EntitySetConstPtr interiorSet = this->DDInterface_->getInterior();
        EntitySetConstPtr interfaceSet = this->DDInterface_->getInterface();

        Array<GO> interiorDofs = this->getEntitySetDofs(interiorSet);
        Array<GO> interfaceDofs = this->getEntitySetDofs(interfaceSet);

        Array<GO> allDofs = Array<GO>(interiorDofs);
        for (UN i = 0; i < interfaceDofs.size(); i++) {
            allDofs.push_back(interfaceDofs[i]);
        }
        std::sort(allDofs.begin(), allDofs.end());

        XMapPtr localMap = MapFactory<LO, GO, NO>::Build(this->K_->getRowMap()->lib(),
                                                         Teuchos::OrdinalTraits<GO>::invalid(),
                                                         allDofs(),
                                                         0,
                                                         this->MpiComm_);
        XMapPtr localSerialMap = MapFactory<LO, GO, NO>::Build(this->K_->getRowMap()->lib(),
                                                               Teuchos::OrdinalTraits<GO>::invalid(),
                                                               allDofs(),
                                                               0,
                                                               this->SerialComm_);

        this->localK = ExtractLocalSubdomainMatrix(this->K_,
                                                   localMap.getConst(),
                                                   localSerialMap.getConst());
    }
}

#endif
