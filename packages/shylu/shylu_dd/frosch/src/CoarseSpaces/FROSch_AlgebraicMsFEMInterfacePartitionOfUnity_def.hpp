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
        this->initializeMaps();
        this->initializeOverlappingMatrices();
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
                    Array<GO> offspringDofs = this->getEntitySetDofs(currEntity->getOffspring());
                    Array<GO> otherDofs(offspringDofs);
                    otherDofs.insert(otherDofs.end(), interiorDofs.begin(), interiorDofs.end());

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
                                   otherDofs(),
                                   kBI);

                    SolverPtr kBBSolver = this->initializeLocalInterfaceSolver(kBB, kBI);

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
                    for (UN k = 0; k < rootsDofs.size(); k++) {
                        kBBSolver->apply(*mVkBV->getVector(k), *mVPhiBV->getVectorNonConst(k));
                    }

                    // If the interface entity is a leaf (no offspring), then
                    // an additional term must be added to Phi_BV so the correct
                    // basis functions are computed.
                    // It can be interpreted as an extension of the BFs values
                    // on the ancestors nodes, e.g. edges to faces in a 3-D
                    // structured domain decomposition.
                    if (offspringDofs.size() == 0) {
                        this->addAncestorExtensionTerm(currEntity,
                                                       currEntityDofs,
                                                       rootsDofs,
                                                       interiorDofs,
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
                        LO rootIdx = roots->getEntity(k)->getRootID();
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
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::initializeMaps() {
        EntitySetConstPtr interiorSet = this->DDInterface_->getInterior();
        EntitySetConstPtr interfaceSet = this->DDInterface_->getInterface();

        Array<GO> interiorDofs = this->getEntitySetDofs(interiorSet);
        Array<GO> interfaceDofs = this->getEntitySetDofs(interfaceSet);
        Array<GO> allDofs = Array<GO>(interiorDofs);
        allDofs.insert(allDofs.end(),
                       interfaceDofs.begin(),
                       interfaceDofs.end());
        std::sort(allDofs.begin(), allDofs.end());

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
                                                                                                                        const XMatrixPtr kIJ) {
        RCP<basic_FancyOStream<char>> blackHoleStream = getFancyOStream(rcp(new oblackholestream()));

        // kIJRowSum = kIJ * 1_I, 1_I = (1 ... 1)
        // This is equivalent to adding the rows of kIJ.
        XMultiVectorPtr kIJRowSum = MultiVectorFactory<SC, LO, GO, NO>::Build(kIJ->getRangeMap(), 1);
        XMultiVectorPtr onesJ = MultiVectorFactory<SC, LO, GO, NO>::Build(kIJ->getDomainMap(), 1);
        onesJ->putScalar(ScalarTraits<SC>::one());
        kIJ->apply(*onesJ, *kIJRowSum);

        // kIIMod = kII + diag(kIJRowSum)
        XMatrixPtr diagKIJ = MatrixFactory<SC, LO, GO, NO>::Build(kIJRowSum->getVector(0));
        XMatrixPtr kIIMod;
        MatrixMatrix<SC, LO, GO, NO>::TwoMatrixAdd(*kII,
                                                   false,
                                                   ScalarTraits<SC>::one(),
                                                   *diagKIJ,
                                                   false,
                                                   ScalarTraits<SC>::one(),
                                                   kIIMod,
                                                   *blackHoleStream,
                                                   false);
        kIIMod->fillComplete();

        // Initialization of the interface solver.
        SolverPtr kIISolver = SolverFactory<SC, LO, GO, NO>::Build(kIIMod,
                                                                   sublist(this->ParameterList_, "InterfaceSolver"),
                                                                   string(""));
        kIISolver->initialize();
        kIISolver->compute();

        return kIISolver;
    }

    template <class SC, class LO, class GO, class NO>
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::addAncestorExtensionTerm(InterfaceEntityPtr entity,
                                                                                           Array<GO>& entityDofs,
                                                                                           Array<GO>& rootsDofs,
                                                                                           Array<GO>& interiorDofs,
                                                                                           RCP<Solver<SC, LO, GO, NO>> kBBSolver,
                                                                                           XMultiVectorPtr mVPhiBV) {
        // Retrieve the dofs of the ancestors of entity.
        EntitySetPtr entityAncestors = entity->getAncestors();
        Array<GO> ancestorsDofs = this->getEntitySetDofs(entityAncestors);

        // Remove the roots' dofs from the ancestors of entity.
        Array<GO> ancestorsDofsNoRoots(ancestorsDofs.size() - rootsDofs.size());
        std::sort(ancestorsDofs.begin(), ancestorsDofs.end());
        std::set_difference(ancestorsDofs.begin(),
                            ancestorsDofs.end(),
                            rootsDofs.begin(),
                            rootsDofs.end(),
                            ancestorsDofsNoRoots.begin());

        Array<GO> remainingDofs(interiorDofs);
        remainingDofs.insert(remainingDofs.end(),
                             entityDofs.begin(),
                             entityDofs.end());
        
        // Assemble the local submatrices.
        // A := ancestors of the entity without the roots
        // R := remaining (entity U interior - ancestors - roots)
        XMatrixPtr kAA, kAR, kAV, kBA;
        BuildSubmatrix(this->localK, ancestorsDofsNoRoots(), kAA);
        BuildSubmatrix(this->localK,
                       ancestorsDofsNoRoots(),
                       rootsDofs(),
                       kAV);
        BuildSubmatrix(this->localK,
                       ancestorsDofsNoRoots(),
                       remainingDofs(),
                       kAR);
        BuildSubmatrix(this->localK,
                       entityDofs(),
                       ancestorsDofsNoRoots(),
                       kBA);
        
        SolverPtr kAASolver = this->initializeLocalInterfaceSolver(kAA, kAR);

        // Convert kAV to a MultiVector so it can be used in kAASolver->apply.
        XMultiVectorPtr mVkAV = MultiVectorFactory<SC, LO, GO, NO>::Build(kAA->getDomainMap(),
                                                                          rootsDofs.size());
        ArrayView<const LO> localColIdx;
        ArrayView<const SC> localVals;
        for (UN k = 0; k < kAV->getLocalNumRows(); k++) {
            kAV->getLocalRowView(k, localColIdx, localVals);
            for (UN l = 0; l < localColIdx.size(); l++) {
                mVkAV->replaceLocalValue(k, localColIdx[l], localVals[l]);
            }
        }

        // mVPhiAV = kAA^-1 * kAV
        XMultiVectorPtr mVPhiAV = MultiVectorFactory<SC, LO, GO, NO>::Build(kAA->getDomainMap(),
                                                                            rootsDofs.size());
        for (UN k = 0; k < rootsDofs.size(); k++) {
            kAASolver->apply(*mVkAV->getVector(k), *mVPhiAV->getVectorNonConst(k));
        }

        // mVPhiBVTmp = kBA * mVPhiAV
        XMultiVectorPtr mVPhiBVTmp = MultiVectorFactory<SC, LO, GO, NO>::Build(kBBSolver->getDomainMap(),
                                                                               rootsDofs.size());
        kBA->apply(*mVPhiAV, *mVPhiBVTmp);

        // mVPhiBVCorr = kBB^-1 * mVPhiBVTmp
        XMultiVectorPtr mVPhiBVCorr = MultiVectorFactory<SC, LO, GO, NO>::Build(kBBSolver->getDomainMap(),
                                                                                rootsDofs.size());
        for (UN k = 0; k < rootsDofs.size(); k++) {
            kBBSolver->apply(*mVPhiBVTmp->getVector(k), *mVPhiBVCorr->getVectorNonConst(k));
        }

        // mVPhiBV += mVPhiBVCorr
        mVPhiBV->update(ScalarTraits<SC>::one(), *mVPhiBVCorr, ScalarTraits<SC>::one());
    }
}

#endif
