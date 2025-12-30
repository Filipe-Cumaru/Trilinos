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

        // Initialization of the interface entities.
        this->DDInterface_->buildEntityHierarchy();
        this->DDInterface_->buildEntityMaps(false,  // vertices
                                            false,  // short edges
                                            false,  // straight edges
                                            false,  // edges
                                            false,  // faces
                                            true,   // roots
                                            false); // leaves
    }

    template <class SC, class LO, class GO, class NO>
    int AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::computePartitionOfUnity(ConstXMultiVectorPtr nodeList) {
        FROSCH_DETAILTIMER_START_LEVELID(computePartitionOfUnityTime,
                                         "AlgebraicMsFEMInterfacePartitionOfUnity::computePartitionOfUnity");

        UN dofsPerNode = this->DDInterface_->getInterface()->getEntity(0)->getDofsPerNode();
        UN numInterfaceDofs = dofsPerNode * this->DDInterface_->getInterface()->getEntity(0)->getNumNodes();

        this->PartitionOfUnityMaps_[0] = this->DDInterface_->getRoots()->getEntityMap();

        // Initialization of a vector to store the IPOU values.
        XMapPtr serialInterfaceMap = MapFactory<LO, GO, NO>::Build(this->DDInterface_->getNodesMap()->lib(),
                                                                   numInterfaceDofs,
                                                                   0,
                                                                   this->SerialComm_);
        this->localIPOUVector = MultiVectorFactory<SC, LO, GO, NO>::Build(serialInterfaceMap,
                                                                          this->DDInterface_->getRoots()->getNumEntities());

        this->Roots_ = this->DDInterface_->getRoots();
        this->Faces_ = this->DDInterface_->getFaces();
        this->Edges_ = this->DDInterface_->getEdges();

        // Initialize some auxiliary structures. Ideally, these should be inside
        // the constructor, but they depend on the classification of the interface
        // entities, which is only done after the constructor call.
        this->initializeDofsArrays();
        this->initializeMaps();
        this->initializeDofIDsArrays();
        this->initializeOverlappingMatrices();
        this->diagInteriorRowSum = this->assembleDiagSumMatrix(this->interiorDofs);
        this->diagFacesRowSum = this->assembleDiagSumMatrix(this->faceDofs);
        
        // Set the IPOU values related to the root dofs to 1.
        for (UN i = 0; i < this->Roots_->getNumEntities(); i++) {
            InterfaceEntityPtr rootEntity = this->Roots_->getEntity(i);
            for (UN j = 0; j < rootEntity->getNumNodes(); j++) {
                for (UN k = 0; k < dofsPerNode; k++) {
                    this->localIPOUVector->replaceLocalValue(rootEntity->getGammaDofID(j, k),
                                                             rootEntity->getRootID(),
                                                             ScalarTraits<SC>::one());
                }
            }
        }

        // Compute the IPOU values for the edge entities.
        for (UN i = 0; i < this->Edges_->getNumEntities(); i++) {
            InterfaceEntityPtr edgeEntity = this->Edges_->getEntity(i);
            UN numRoots = edgeEntity->getRoots()->getNumEntities();

            // Skip if root.
            LO rootId = edgeEntity->getRootID();
            if (rootId != -1) {
                continue;
            }

            FROSCH_ASSERT(numRoots != 0, "rootID==-1 but numRoots==0!");

            this->computeEntityIPOU(edgeEntity, true, false);
        }

        // Compute the IPOU values for the face entities.
        for (UN i = 0; i < this->Faces_->getNumEntities(); i++) {
            InterfaceEntityPtr faceEntity = this->Faces_->getEntity(i);
            UN numRoots = faceEntity->getRoots()->getNumEntities();

            // Skip if root.
            LO rootId = faceEntity->getRootID();
            if (rootId != -1) {
                continue;
            }

            FROSCH_ASSERT(numRoots != 0, "rootID==-1 but numRoots==0!");

            this->computeEntityIPOU(faceEntity, false, true);
        }

        this->LocalPartitionOfUnity_[0] = this->localIPOUVector.getConst();

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
        this->rootDofs = this->getEntitySetDofs(this->Roots_);
        std::sort(this->rootDofs.begin(), this->rootDofs.end());

        // Interior entities and their dofs.
        EntitySetConstPtr interiorSet = this->DDInterface_->getInterior();
        this->interiorDofs = this->getEntitySetDofs(interiorSet);

        // Face entities and their dofs.
        Array<GO> faceEntityDofs;
        for (UN i = 0; i < this->Faces_->getNumEntities(); i++) {
            InterfaceEntityPtr faceEntity = this->Faces_->getEntity(i);
            LO rootId = faceEntity->getRootID();
            if (rootId != -1) {
                continue;
            }
            faceEntityDofs = this->getEntityDofs(faceEntity);
            this->faceDofs.insert(this->faceDofs.end(),
                                  faceEntityDofs.begin(),
                                  faceEntityDofs.end());
        }
        std::sort(this->faceDofs.begin(), this->faceDofs.end());

        // Edge entities and their dofs.
        Array<GO> edgeEntityDofs;
        for (UN i = 0; i < this->Edges_->getNumEntities(); i++) {
            InterfaceEntityPtr edgeEntity = this->Edges_->getEntity(i);
            LO rootId = edgeEntity->getRootID();
            if (rootId != -1) {
                continue;
            }
            edgeEntityDofs = this->getEntityDofs(edgeEntity);
            this->edgeDofs.insert(this->edgeDofs.end(),
                                  edgeEntityDofs.begin(),
                                  edgeEntityDofs.end());
        }
        std::sort(this->edgeDofs.begin(), this->edgeDofs.end());

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
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::initializeDofIDsArrays() {
        this->gammaDofIDs = Array<LO>(this->repeatedMap->getLocalNumElements(), -1);
        this->rootIDs = Array<LO>(this->repeatedMap->getLocalNumElements(), -1);
        EntitySetPtrVecPtr entitySetVector = this->DDInterface_->getEntitySetVector();
        for (UN i = 0; i < entitySetVector.size(); i++) {
            for (UN j = 0; j < entitySetVector[i]->getNumEntities(); j++) {
                InterfaceEntityPtr entity = entitySetVector[i]->getEntity(j);
                for (UN k = 0; k < entity->getNumNodes(); k++) {
                    for (UN l = 0; l < entity->getDofsPerNode(); l++) {
                        LO localDofID = this->repeatedMap->getLocalElement(entity->getGlobalDofID(k, l));
                        this->gammaDofIDs[localDofID] = entity->getGammaDofID(k, l);
                        this->rootIDs[localDofID] = entity->getRootID();
                    }
                }
            }
        }
    }

    template <class SC, class LO, class GO, class NO>
    RCP<const Matrix<SC,LO,GO,NO>> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::assembleDiagSumMatrix(const Array<GO>& colIndices) const {
        // Compute the row sum and convert the resulting vector in a diagonal matrix.
        XMultiVectorPtr rowSum = sumMatrixRows<SC, LO, GO, NO>(this->overlappingK,
                                                               colIndices);
        XMatrixPtr diagRowSum = MatrixFactory<SC, LO, GO, NO>::Build(rowSum->getVector(0));

        // Convert the resulting matrix into a local (serial) matrix.
        ConstXMatrixPtr serialDiagRowSum = ExtractLocalSubdomainMatrix(diagRowSum.getConst(),
                                                                       this->repeatedMap.getConst(),
                                                                       this->serialRepeatedMap.getConst());
        return serialDiagRowSum;
    }

    template <class SC, class LO, class GO, class NO>
    RCP<Solver<SC, LO, GO, NO>> AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::initializeLocalInterfaceSolver(const XMatrixPtr kII,
                                                                                                                        const XMatrixPtr diagSumInterior,
                                                                                                                        const XMatrixPtr diagSumExtra) const {
        // Compute kIIMod = kII + diagSumInterior
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

        // If diagSumExtra was provided, compute kIIMod += diagSumExtra.
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
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::computeEntityIPOU(const InterfaceEntityPtr entity,
                                                                                    bool removeFacesFromDiag,
                                                                                    bool addAncestorTerm) const {
        UN dofsPerNode = this->DDInterface_->getInterface()->getEntity(0)->getDofsPerNode();

        // Retrieve the dofs of the current entity and of its roots.
        EntitySetPtr entityRoots = entity->getRoots();
        UN numRoots = entityRoots->getNumEntities();
        Array<GO> entityRootsDofs = this->getEntitySetDofs(entityRoots);
        Array<GO> entityDofs = this->getEntityDofs(entity);

        // Exract the submatrices required to assemble the IPOU.
        // kBB is the block related to the dofs in `entity`, and
        // kBV is the block coupling the dofs in `entity` and its roots.
        XMatrixPtr kBB;
        BuildSubmatrix(this->localK, entityDofs(), kBB);
        XMatrixPtr kBV;
        BuildSubmatrix(this->localK,
                        entityDofs(),
                        entityRootsDofs(),
                        kBV);

        // Extract the submatrices of the global row sum diagonal matrices.
        // For any non-root entity, the influence of the interior dofs must
        // be removed.
        XMatrixPtr diagInteriorRowSumBlock;
        BuildSubmatrix(this->diagInteriorRowSum,
                       entityDofs(),
                       diagInteriorRowSumBlock);

        // If necessary, remove the influence of the face dofs (e.g., for edge entities).
        XMatrixPtr diagFacesRowSumBlock = null;
        if (removeFacesFromDiag) {
            BuildSubmatrix(this->diagFacesRowSum,
                           entityDofs(),
                           diagFacesRowSumBlock);
        }

        // Initialize a solver object to solve the reduced boundary condition
        // system (kBB + diagInteriorRowSumBlock + diagFacesRowSumBlock)x = b.
        SolverPtr kBBSolver = this->initializeLocalInterfaceSolver(kBB,
                                                                   diagInteriorRowSumBlock,
                                                                   diagFacesRowSumBlock);

        // Apply kBBSolver to the columns of kBV.
        XMultiVectorPtr mVkBV = matrixToMultiVector<SC, LO, GO, NO>(kBV);
        XMultiVectorPtr mVPhiBV = MultiVectorFactory<SC, LO, GO, NO>::Build(kBB->getDomainMap(),
                                                                            entityRootsDofs.size());
        for (UN i = 0; i < entityRootsDofs.size(); i++) {
            kBBSolver->apply(*mVkBV->getVector(i),
                             *mVPhiBV->getVectorNonConst(i));
        }

        // For face entities in 3D, an additional IPOU term related to the
        // ancestors of the entitty must be added.
        if (addAncestorTerm) {
            EntitySetPtr entityAncestors = entity->getAncestors();
            Array<GO> ancestorDofs = this->getEntitySetDofs(entityAncestors);
            Array<GO> ancestorDofsNoRoots(ancestorDofs.size());
            auto it = std::set_difference(ancestorDofs.begin(),
                                          ancestorDofs.end(),
                                          entityRootsDofs.begin(),
                                          entityRootsDofs.end(),
                                          ancestorDofsNoRoots.begin());
            ancestorDofsNoRoots.resize(it - ancestorDofsNoRoots.begin());
            if (ancestorDofsNoRoots.size() > 0) {
                this->addAncestorTerm(entity,
                                      entityDofs,
                                      entityRootsDofs,
                                      ancestorDofsNoRoots,
                                      kBBSolver,
                                      mVPhiBV);
            }
        }

        // Normalize the computed IPOU values and store them in `ipouVector`.
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
        for (UN i = 0; i < numRoots; i++) {
            LO rootIdx = entityRoots->getEntity(i)->getRootID();
            ArrayRCP<const SC> mVPhiBVk = mVPhiBV->getData(i);
            UN n = 0;
            for (UN j = 0; j < entity->getNumNodes(); j++) {
                for (UN k = 0; k < dofsPerNode; k++) {
                    SC value = mVPhiBVk[n] / sumPhiBV[n];
                    this->localIPOUVector->replaceLocalValue(entity->getGammaDofID(j, k),
                                                             rootIdx,
                                                             value);
                    n += 1;
                }
            }
        }
    }

    template <class SC, class LO, class GO, class NO>
    void AlgebraicMsFEMInterfacePartitionOfUnity<SC, LO, GO, NO>::addAncestorTerm(const InterfaceEntityPtr entity,
                                                                                  Array<GO> entityDofs,
                                                                                  Array<GO> entityRootsDofs,
                                                                                  Array<GO> ancestorDofsNoRoots,
                                                                                  const SolverPtr kBBSolver,
                                                                                  XMultiVectorPtr mVPhiBV) const {
        // Extract the blocks of the system matrix K related to the ancestors
        // and the entity.
        // A -> ancestors
        // B -> current entity
        // V -> roots
        XMatrixPtr kBA;
        BuildSubmatrix(this->localK,
                       entityDofs(),
                       ancestorDofsNoRoots(),
                       kBA);

        // Initialize mVPhiAV with the values of the IPOU on the ancestors.
        XMultiVectorPtr mVPhiAV = MultiVectorFactory<SC, LO, GO, NO>::Build(kBA->getDomainMap(),
                                                                            entityRootsDofs.size());
        for (UN i = 0; i < entityRootsDofs.size(); i++) {
            LO rootDofLocalIdx = this->repeatedMap->getLocalElement(entityRootsDofs[i]);
            LO rootID = this->rootIDs[rootDofLocalIdx];
            ArrayRCP<const SC> rootIpou = this->localIPOUVector->getData(rootID);
            for (UN j = 0; j < ancestorDofsNoRoots.size(); j++) {
                LO nonRootAncestorDofLocalIdx = this->repeatedMap->getLocalElement(ancestorDofsNoRoots[j]);
                LO gammaDofID = this->gammaDofIDs[nonRootAncestorDofLocalIdx];
                LO localRowIdx = kBA->getDomainMap()->getLocalElement(ancestorDofsNoRoots[j]);
                mVPhiAV->replaceLocalValue(localRowIdx, i, rootIpou[gammaDofID]);
            }
        }

        // mVPhiBVTmp = kBA * mVPhiAV
        XMultiVectorPtr mVPhiBVTmp = MultiVectorFactory<SC, LO, GO, NO>::Build(kBBSolver->getDomainMap(),
                                                                               entityRootsDofs.size());
        for (UN k = 0; k < entityRootsDofs.size(); k++) {
            kBA->apply(*mVPhiAV->getVector(k),
                       *mVPhiBVTmp->getVectorNonConst(k));
        }

        // mVPhiBVCorr = kBBMod^-1 * mVPhiBVTmp
        XMultiVectorPtr mVPhiBVCorr = MultiVectorFactory<SC, LO, GO, NO>::Build(kBBSolver->getDomainMap(),
                                                                                entityRootsDofs.size());
        for (UN k = 0; k < entityRootsDofs.size(); k++) {
            kBBSolver->apply(*mVPhiBVTmp->getVector(k),
                             *mVPhiBVCorr->getVectorNonConst(k));
        }

        // mVPhiBV -= mVPhiBVCorr
        mVPhiBV->update(-ScalarTraits<SC>::one(),
                        *mVPhiBVCorr,
                        ScalarTraits<SC>::one());
    }
}

#endif
