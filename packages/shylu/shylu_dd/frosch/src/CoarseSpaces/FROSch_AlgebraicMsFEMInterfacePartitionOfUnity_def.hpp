#ifndef _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DEF_HPP
#define _FROSCH_ALGEBRAICMSFEMINTERFACEPARTITIONOFUNITY_DEF_HPP

#include <FROSch_AlgebraicMsFEMInterfacePartitionOfUnity_decl.hpp>
#include <FROSch_SolverFactory_def.hpp>
#include <FROSch_Tools_def.hpp>
#include <Xpetra_MatrixMatrix_def.hpp>
#include <algorithm>

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
        this->blackHoleStream = getFancyOStream(rcp(new oblackholestream()));

        this->DDInterface_->buildEntityHierarchy();
        this->DDInterface_->buildEntityMaps(false,  // vertices
                                            false,  // short edges
                                            false,  // straight edges
                                            false,  // edges
                                            false,  // faces
                                            true,   // roots
                                            false); // leaves

        this->initializeDofsArrays();
        this->initializeMaps();
        this->initializeOverlappingMatrices();

        this->diagInteriorRowSum = this->assembleDiagSumMatrix(this->interiorDofs);
        this->diagLeavesRowSum = this->assembleDiagSumMatrix(this->leafDofs);
    }

    template <class SC, class LO, class GO, class NO>
    int AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::computePartitionOfUnity(ConstXMultiVectorPtr nodeList) {
        FROSCH_DETAILTIMER_START_LEVELID(computePartitionOfUnityTime,
                                         "AlgebraicMsFEMInterfacePartitionOfUnity::computePartitionOfUnity");

        UN dofsPerNode = this->DDInterface_->getInterface()->getEntity(0)->getDofsPerNode();
        UN numInterfaceDofs = dofsPerNode * this->DDInterface_->getInterface()->getEntity(0)->getNumNodes();
        UN numInterfaceNotRootNotLeafDofs = numInterfaceDofs - this->rootDofs.size() - this->leafDofs.size();

        EntitySetPtrVecPtr entitySetVector = this->DDInterface_->getEntitySetVector();
        this->PartitionOfUnityMaps_[0] = this->DDInterface_->getRoots()->getEntityMap();

        // Initialization of a vector to store the IPOU values.
        XMapPtr serialInterfaceMap = MapFactory<LO, GO, NO>::Build(this->DDInterface_->getNodesMap()->lib(),
                                                                   numInterfaceDofs,
                                                                   0,
                                                                   this->SerialComm_);
        XMultiVectorPtr ipouVector = MultiVectorFactory<SC, LO, GO, NO>::Build(serialInterfaceMap,
                                                                               this->DDInterface_->getRoots()->getNumEntities());

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

                    EntitySetPtr currEntityRoots = currEntity->getRoots();
                    Array<GO> currEntityRootsDofs = this->getEntitySetDofs(currEntityRoots);
                    Array<GO> currEntityDofs = this->getEntityDofs(currEntity);
                    Array<GO> offspringDofs = this->getEntitySetDofs(currEntity->getOffspring());
                    Array<GO> otherDofs(offspringDofs);
                    otherDofs.insert(otherDofs.end(),
                                     this->interiorDofs.begin(),
                                     this->interiorDofs.end());

                    // Exract the submatrices required to assemble the IPOU.
                    // These are equivalent to the blocks in the wirebasket order:
                    //           (kII kIB kIV)
                    // W^T K W = (kBI kBB kBV)
                    //           (kVI kVB kVV)
                    XMatrixPtr kBB;
                    XMatrixPtr kBV;
                    XMatrixPtr diagInteriorRowSumBlock;
                    BuildSubmatrix(this->localK, currEntityDofs(), kBB);
                    BuildSubmatrix(this->diagInteriorRowSum,
                                   currEntityDofs(),
                                   diagInteriorRowSumBlock);
                    BuildSubmatrix(this->localK,
                                   currEntityDofs(),
                                   currEntityRootsDofs(),
                                   kBV);

                    SolverPtr kBBSolver;
                    if (numInterfaceNotRootNotLeafDofs != 0 && offspringDofs.size() != 0) {
                        XMatrixPtr diagLeavesRowSumBlock;
                        BuildSubmatrix(this->diagLeavesRowSum,
                                       currEntityDofs(),
                                       diagLeavesRowSumBlock);
                        kBBSolver = this->initializeLocalInterfaceSolver(kBB,
                                                                         diagInteriorRowSumBlock,
                                                                         diagLeavesRowSumBlock);
                    } else {
                        kBBSolver = this->initializeLocalInterfaceSolver(kBB,
                                                                         diagInteriorRowSumBlock);
                    }

                    // Convert kBV to a MultiVector so it can be used in
                    // kBBSolver->apply(...).
                    XMultiVectorPtr mVkBV = matrixToMultiVector<SC, LO, GO, NO>(kBV);

                    // Compute the solution on the interface.
                    XMultiVectorPtr mVPhiBV = MultiVectorFactory<SC, LO, GO, NO>::Build(kBB->getDomainMap(),
                                                                                        currEntityRootsDofs.size());
                    for (UN k = 0; k < currEntityRootsDofs.size(); k++) {
                        kBBSolver->apply(*mVkBV->getVector(k),
                                         *mVPhiBV->getVectorNonConst(k));
                    }

                    // If the interface entity is a leaf (no offspring), then
                    // an additional term must be added to Phi_BV so the correct
                    // basis functions are computed.
                    // It can be interpreted as an extension of the BFs values
                    // on the ancestors nodes, e.g. edges to faces in a 3-D
                    // structured domain decomposition.
                    if (numInterfaceNotRootNotLeafDofs != 0 && offspringDofs.size() == 0) {
                        this->addAncestorTerm(currEntity,
                                              currEntityDofs,
                                              currEntityRootsDofs,
                                              kBBSolver,
                                              mVPhiBV);
                    }

                    // Add up the interface values to get a POU. This is done by multiplying
                    // mVPhiBV by a vector of ones (onesV).
                    XMultiVectorPtr onesV = MultiVectorFactory<SC, LO, GO, NO>::Build(kBV->getDomainMap(), 1);
                    XMultiVectorPtr mVSumPhiBV = MultiVectorFactory<SC, LO, GO, NO>::Build(kBV->getRangeMap(), 1);
                    onesV->putScalar(ScalarTraits<SC>::one());
                    mVSumPhiBV->multiply(Teuchos::ETransp::NO_TRANS,
                                         Teuchos::ETransp::NO_TRANS,
                                         ScalarTraits<SC>::one(),
                                         *mVPhiBV,
                                         *onesV,
                                         ScalarTraits<SC>::zero());
                    ArrayRCP<const SC> sumPhiBV = mVSumPhiBV->getData(0);

                    // Set the entries on the IPOU vector.
                    for (UN k = 0; k < numRoots; k++) {
                        LO rootIdx = currEntityRoots->getEntity(k)->getRootID();
                        ArrayRCP<const SC> mVPhiBVk = mVPhiBV->getData(k);
                        UN n = 0;
                        for (UN l = 0; l < currEntity->getNumNodes(); l++) {
                            for (UN m = 0; m < dofsPerNode; m++) {
                                SC value = mVPhiBVk[n] / sumPhiBV[n];
                                ipouVector->replaceLocalValue(currEntity->getGammaDofID(l, m),
                                                              rootIdx,
                                                              value * ScalarTraits<SC>::one());
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
    Array<GO> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::getEntityDofs(InterfaceEntityPtr entity) const {
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
    Array<GO> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::getEntitySetDofs(EntitySetConstPtr entitySet) const {
        Array<GO> entitySetDofs;
        for (UN i = 0; i < entitySet->getNumEntities(); i++) {
            InterfaceEntityPtr tmpEntity = entitySet->getEntity(i);
            Array<GO> entityDofs = this->getEntityDofs(tmpEntity);
            entitySetDofs.insert(entitySetDofs.end(),
                                 entityDofs.begin(),
                                 entityDofs.end());
        }
        return entitySetDofs;
    }

    template <class SC, class LO, class GO, class NO>
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::initializeOverlappingMatrices() {
        XMatrixPtr nonConstOverlappingK = MatrixFactory<SC, LO, GO, NO>::Build(this->repeatedMap,
                                                                               2 * this->K_->getGlobalMaxNumRowEntries());
        RCP<Import<LO, GO, NO>> scatter = ImportFactory<LO, GO, NO>::Build(this->K_->getRowMap(),
                                                                           this->repeatedMap);
        nonConstOverlappingK->doImport(*this->K_, *scatter, ADD);
        nonConstOverlappingK->fillComplete();
        this->overlappingK = nonConstOverlappingK.getConst();

        this->localK = ExtractLocalSubdomainMatrix(this->K_,
                                                   this->repeatedMap.getConst(),
                                                   this->serialRepeatedMap.getConst());
    }

    template <class SC, class LO, class GO, class NO>
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::initializeDofsArrays() {
        // Retrieve all root entities owned by the process.
        EntitySetPtr rootsSet = this->DDInterface_->getRoots();
        this->rootDofs = this->getEntitySetDofs(rootsSet);
        std::sort(this->rootDofs.begin(), this->rootDofs.end());

        // Leaf entities and dofs.
        EntitySetConstPtr leavesSet = this->DDInterface_->getLeafs();
        this->leafDofs = this->getEntitySetDofs(leavesSet);
        std::sort(this->leafDofs.begin(), this->leafDofs.end());

        // Interior entities and their dofs.
        EntitySetConstPtr interiorSet = this->DDInterface_->getInterior();
        this->interiorDofs = this->getEntitySetDofs(interiorSet);

        // All interface entities owned by the process.
        EntitySetConstPtr interfaceSet = this->DDInterface_->getInterface();
        this->interfaceDofs = this->getEntitySetDofs(interfaceSet);
    }

    template <class SC, class LO, class GO, class NO>
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::initializeMaps() {
        Array<GO> allDofs = Array<GO>(this->interiorDofs.size() + this->interfaceDofs.size());
        std::set_union(this->interiorDofs.begin(), this->interiorDofs.end(),
                       this->interfaceDofs.begin(), this->interfaceDofs.end(),
                       allDofs.begin());
        this->repeatedMap = MapFactory<LO, GO, NO>::Build(this->K_->getRowMap()->lib(),
                                                         Teuchos::OrdinalTraits<GO>::invalid(),
                                                         allDofs(),
                                                         0,
                                                         this->MpiComm_);
        this->serialRepeatedMap = MapFactory<LO, GO, NO>::Build(this->K_->getRowMap()->lib(),
                                                               Teuchos::OrdinalTraits<GO>::invalid(),
                                                               allDofs(),
                                                               0,
                                                               this->SerialComm_);
    }

    template <class SC, class LO, class GO, class NO>
    RCP<const Matrix<SC,LO,GO,NO>> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::assembleDiagSumMatrix(const Array<GO>& colIndices) const {
        XMultiVectorPtr rowSum = sumMatrixRows<SC, LO, GO, NO>(this->overlappingK,
                                                               colIndices);
        XMatrixPtr diagRowSum = MatrixFactory<SC, LO, GO, NO>::Build(rowSum->getVector(0));
        ConstXMatrixPtr serialDiagRowSum = ExtractLocalSubdomainMatrix(diagRowSum.getConst(),
                                                                       this->repeatedMap.getConst(),
                                                                       this->serialRepeatedMap.getConst());
        return serialDiagRowSum;
    }

    template <class SC, class LO, class GO, class NO>
    RCP<Solver<SC, LO, GO, NO>> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::initializeLocalInterfaceSolver(const XMatrixPtr kII,
                                                                                                                        const XMatrixPtr diagSumInterior,
                                                                                                                        const XMatrixPtr diagSumExtra) const {
        XMatrixPtr kIIMod;
        MatrixMatrix<SC, LO, GO, NO>::TwoMatrixAdd(*kII,
                                                   false,
                                                   ScalarTraits<SC>::one(),
                                                   *diagSumInterior,
                                                   false,
                                                   ScalarTraits<SC>::one(),
                                                   kIIMod,
                                                   *this->blackHoleStream,
                                                   false);
        kIIMod->fillComplete();

        if (!diagSumExtra.is_null()) {
            XMatrixPtr kIIModExtra;
            MatrixMatrix<SC, LO, GO, NO>::TwoMatrixAdd(*diagSumExtra,
                                                       false,
                                                       ScalarTraits<SC>::one(),
                                                       *kIIMod,
                                                       false,
                                                       ScalarTraits<SC>::one(),
                                                       kIIModExtra,
                                                       *this->blackHoleStream,
                                                       false);
            kIIModExtra->fillComplete();
            kIIMod = kIIModExtra;
        }

        // Initialization of the interface solver.
        SolverPtr kIISolver = SolverFactory<SC, LO, GO, NO>::Build(kIIMod,
                                                                   sublist(this->ParameterList_, "InterfaceSolver"),
                                                                   string(""));
        kIISolver->initialize();
        kIISolver->compute();

        return kIISolver;
    }

    template <class SC, class LO, class GO, class NO>
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::addAncestorTerm(const InterfaceEntityPtr entity,
                                                                                  Array<GO> entityDofs,
                                                                                  Array<GO> entityRootsDofs,
                                                                                  const SolverPtr kBBSolver,
                                                                                  XMultiVectorPtr mVPhiBV) const {
        EntitySetPtr ancestors = entity->getAncestors();
        Array<GO> ancestorDofs = this->getEntitySetDofs(ancestors);
        Array<GO> ancestorDofsNoRoots(ancestorDofs.size() - entityRootsDofs.size());
        std::set_difference(ancestorDofs.begin(),
                            ancestorDofs.end(),
                            entityRootsDofs.begin(),
                            entityRootsDofs.end(),
                            ancestorDofsNoRoots.begin());
        
        XMatrixPtr kAA;
        XMatrixPtr kBA;
        XMatrixPtr kAV;
        BuildSubmatrix(this->localK, ancestorDofsNoRoots(), kAA);
        BuildSubmatrix(this->localK,
                       entityDofs(),
                       ancestorDofsNoRoots(),
                       kBA);
        BuildSubmatrix(this->localK,
                       ancestorDofsNoRoots(),
                       entityRootsDofs(),
                       kAV);
        
        XMatrixPtr diagInteriorRowSumAncestorBlock;
        XMatrixPtr diagLeavesRowSumAncestorBlock;
        BuildSubmatrix(this->diagInteriorRowSum,
                       ancestorDofsNoRoots(),
                       diagInteriorRowSumAncestorBlock);
        BuildSubmatrix(this->diagLeavesRowSum,
                       ancestorDofsNoRoots(),
                       diagLeavesRowSumAncestorBlock);

        SolverPtr kAASolver = this->initializeLocalInterfaceSolver(kAA,
                                                                   diagInteriorRowSumAncestorBlock,
                                                                   diagLeavesRowSumAncestorBlock);
        
        XMultiVectorPtr mVkAV = matrixToMultiVector<SC, LO, GO, NO>(kAV);

        XMultiVectorPtr mVPhiAV = MultiVectorFactory<SC, LO, GO, NO>::Build(kAA->getDomainMap(),
                                                                            entityRootsDofs.size());
        for (UN k = 0; k < entityRootsDofs.size(); k++) {
            kAASolver->apply(*mVkAV->getVector(k),
                             *mVPhiAV->getVectorNonConst(k));
        }

        XMultiVectorPtr mVPhiBVTmp = MultiVectorFactory<SC, LO, GO, NO>::Build(kBBSolver->getDomainMap(),
                                                                               entityRootsDofs.size());
        for (UN k = 0; k < entityRootsDofs.size(); k++) {
            kBA->apply(*mVPhiAV->getVector(k),
                       *mVPhiBVTmp->getVectorNonConst(k));
        }

        XMultiVectorPtr mVPhiBVCorr = MultiVectorFactory<SC, LO, GO, NO>::Build(kBBSolver->getDomainMap(),
                                                                                entityRootsDofs.size());
        for (UN k = 0; k < entityRootsDofs.size(); k++) {
            kBBSolver->apply(*mVPhiBVTmp->getVector(k),
                             *mVPhiBVCorr->getVectorNonConst(k));
        }

        mVPhiBV->update(-ScalarTraits<SC>::one(),
                        *mVPhiBVCorr,
                        ScalarTraits<SC>::one());
    }
}

#endif
