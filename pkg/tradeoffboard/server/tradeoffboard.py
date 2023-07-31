from itertools import chain, combinations
import logging, random

# See the formalization in LaTeX for more details 
# on how the optimization algorithm works


# power set function
def powerset(iterable):
    s = list(iterable)
    return set(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))




# ==== configuration (ideallly coming from a configuration file, now hardcoded for ease of use) ====

# prefilters
prefilters = [
    frozenset(['cryptoac-dm', 'vehicle1']),
    frozenset(['cryptoac-mm', 'vehicle1']),

    frozenset(['cryptoac-dm', 'vehicle2']),
    frozenset(['cryptoac-mm', 'vehicle2']),

    frozenset(['front-end-pub', 'public']),
    frozenset(['front-end-pub', 'edge']),
    frozenset(['front-end-pub', 'private']),
    frozenset(['front-end-pub', 'vehicle2']),

    frozenset(['front-end-sub', 'public']),
    frozenset(['front-end-sub', 'edge']),
    frozenset(['front-end-sub', 'private']),
    frozenset(['front-end-sub', 'vehicle1']),
]

# entity -> natural number (return the min/max replicas allowed of the given entity)
def computeReplicas(entity):
    replicas = {
        frozenset(['cryptoac-proxy-a']): (1, 1),
        frozenset(['cryptoac-proxy-b']): (1, 1),
        frozenset(['cryptoac-mm']): (1, 1),
        frozenset(['cryptoac-dm']): (1, 1),
        frozenset(['front-end-pub']): (1, 1),
        frozenset(['front-end-sub']): (1, 1)
    }
    return replicas[frozenset([entity])]

# pointsOfAttack -> likelihoods (return the likelihood that an attacker strikes in a given point of attack)
def computeLikelihood(pointOfAttack):
    likelihood = {
        frozenset(['public', 'edge']): "Medium",
        frozenset(['public', 'private']): "Low",
        frozenset(['public', 'vehicle1']): "High",
        frozenset(['public', 'vehicle2']): "High",
        frozenset(['edge', 'private']): "Low",
        frozenset(['edge', 'vehicle1']): "High",
        frozenset(['edge', 'vehicle2']): "High",
        frozenset(['private', 'vehicle1']): "High",
        frozenset(['private', 'vehicle2']): "High",
        frozenset(['vehicle1', 'vehicle2']): "High",
        'public': "High",
        'edge': "High",
        'private': "Low",
        'vehicle1': "Low",
        'vehicle2': "Low",
    }
    return likelihood[pointOfAttack]

# properties -> natural number (return the weight of the given property)
weightsAreDefined = True
def computeWeight(property):
    weights = {
        frozenset(['C']): 2,
        frozenset(['I']): 2,
        frozenset(['A']): 1
    }
    return weights[frozenset(property)]






# ==== functions ====

# impacts \times likelihoods -> risks (return the risk level for the given impact and likelihood)
def computeRiskLevel(impact, likelihood):
    riskTable = {
        frozenset(["None", "None"]): "None",
        frozenset(["None", "Low"]): "None",
        frozenset(["None", "Medium"]): "None",
        frozenset(["None", "High"]): "None",

        frozenset(["Low", "None"]): "None",
        frozenset(["Low", "Low"]): "Low",
        frozenset(["Low", "Medium"]): "Low",
        frozenset(["Low", "High"]): "Low",

        frozenset(["Medium", "None"]): "None",
        frozenset(["Medium", "Low"]): "Low",
        frozenset(["Medium", "Medium"]): "Medium",
        frozenset(["Medium", "High"]): "Medium",
        
        frozenset(["High", "None"]): "None",
        frozenset(["High", "Low"]): "Low",
        frozenset(["High", "Medium"]): "Medium",
        frozenset(["High", "High"]): "High",
    }
    return riskTable[frozenset([impact, likelihood])]

# resources \times properties -> impacts (return the impact on the given property (asset) when the given resource is compromised)
def computeResourcesAndImpact(resource, property):
    resourceAndImpact = {
        frozenset(["Data", "C"]): "High",
        frozenset(["Data", "I"]): "High",
        frozenset(["Data", "A"]): "None",
        frozenset(["Encrypted Data", "C"]): "None",
        frozenset(["Encrypted Data", "I"]): "Low",
        frozenset(["Encrypted Data", "A"]): "Medium",
        frozenset(["Secret Keys", "C"]): "High",
        frozenset(["Secret Keys", "I"]): "None",
        frozenset(["Secret Keys", "A"]): "None",
        frozenset(["Metadata", "C"]): "None",
        frozenset(["Metadata", "I"]): "High",
        frozenset(["Metadata", "A"]): "Medium",
    }
    return resourceAndImpact[frozenset([resource, property])]

# targets \times properties -> powerSet(resources) (return which resources an attacker compromising the given property in the given target can threaten)
def computeTargetAndProperties(target, property):
    targetsAndProperties = {
        frozenset(['cryptoac-proxy-a', "C"]): frozenset(["Encrypted Data", "Secret Keys", "Metadata"]),
        frozenset(['cryptoac-proxy-b', "C"]): frozenset(["Encrypted Data", "Secret Keys", "Metadata"]),        
        frozenset(['cryptoac-mm', "C"]): frozenset(["Metadata"]),
        frozenset(['cryptoac-rm', "C"]): frozenset(["Encrypted Data", "Metadata"]),
        frozenset(['cryptoac-dm', "C"]): frozenset(["Encrypted Data"]),
        frozenset(['front-end-pub', "C"]): frozenset(["Data"]),
        frozenset(['front-end-sub', "C"]): frozenset(["Data"]),

        frozenset([frozenset(['cryptoac-proxy-a', 'front-end-pub']), "C"]): frozenset(["Data"]),
        frozenset([frozenset(['cryptoac-proxy-b', 'front-end-sub']), "C"]): frozenset(["Data"]),
        frozenset([frozenset(['cryptoac-proxy-a', 'cryptoac-mm']), "C"]): frozenset(["Metadata"]),
        frozenset([frozenset(['cryptoac-proxy-a', 'cryptoac-dm']), "C"]): frozenset(["Encrypted Data"]),
        frozenset([frozenset(['cryptoac-proxy-b', 'cryptoac-mm']), "C"]): frozenset(["Metadata"]),
        frozenset([frozenset(['cryptoac-proxy-b', 'cryptoac-dm']), "C"]): frozenset(["Encrypted Data"]),
        frozenset([frozenset(['cryptoac-rm', 'cryptoac-mm']), "C"]): frozenset(["Metadata"]),
        frozenset([frozenset(['cryptoac-rm', 'cryptoac-dm']), "C"]): frozenset(["Encrypted Data"]),


        frozenset(['cryptoac-proxy-a', "I"]): frozenset(["Encrypted Data", "Secret Keys", "Metadata"]),
        frozenset(['cryptoac-proxy-b', "I"]): frozenset(["Encrypted Data", "Secret Keys", "Metadata"]),
        frozenset(['cryptoac-mm', "I"]): frozenset(["Metadata"]),
        frozenset(['cryptoac-rm', "I"]): frozenset(["Encrypted Data", "Metadata"]),
        frozenset(['cryptoac-dm', "I"]): frozenset(["Encrypted Data"]),
        frozenset(['front-end-pub', "I"]): frozenset(["Data"]),
        frozenset(['front-end-sub', "I"]): frozenset(["Data"]),

        frozenset([frozenset(['cryptoac-proxy-a', 'front-end-pub']), "I"]): frozenset(["Data"]),
        frozenset([frozenset(['cryptoac-proxy-b', 'front-end-sub']), "I"]): frozenset(["Data"]),
        frozenset([frozenset(['cryptoac-proxy-a', 'cryptoac-mm']), "I"]): frozenset(["Metadata"]),
        frozenset([frozenset(['cryptoac-proxy-a', 'cryptoac-dm']), "I"]): frozenset(["Encrypted Data"]),
        frozenset([frozenset(['cryptoac-proxy-b', 'cryptoac-mm']), "I"]): frozenset(["Metadata"]),
        frozenset([frozenset(['cryptoac-proxy-b', 'cryptoac-dm']), "I"]): frozenset(["Encrypted Data"]),
        frozenset([frozenset(['cryptoac-rm', 'cryptoac-mm']), "I"]): frozenset(["Metadata"]),
        frozenset([frozenset(['cryptoac-rm', 'cryptoac-dm']), "I"]): frozenset(["Encrypted Data"]),


        frozenset(['cryptoac-proxy-a', "A"]): frozenset(["Secret Keys"]),
        frozenset(['cryptoac-proxy-b', "A"]): frozenset(["Secret Keys"]),
        frozenset(['front-end-pub', "A"]): frozenset(["Data"]),
        frozenset(['front-end-sub', "A"]): frozenset(["Data"]),
        frozenset(['cryptoac-mm', "A"]): frozenset(["Metadata"]),
        frozenset(['cryptoac-dm', "A"]): frozenset(["Encrypted Data"])
    }

    #logging.debug("[TRADEOFFBOARD] asking whether " + str(frozenset([target, property])) + ": " + str(frozenset([target, property]) in targetsAndProperties.keys()) )
    if (frozenset([target, property]) in targetsAndProperties.keys()):
        return targetsAndProperties[frozenset([target, property])]
    else:
        return set()

# function: targets \times properties -> impacts (return the impact on the given property (asset) that we have when the given target is compromised)
def computeTargetsAndImpact(target, property):
    
    resources = computeTargetAndProperties(target, property)
    i = "None"
    for resource in resources:
        j = computeResourcesAndImpact(resource, property)
        if (j == "High"):
            i = j
        elif (j == "Medium" and i != "High"):
            i = j
        elif(j == "Low" and i != "High" and i != "Medium"):
            i = j
    return i

# function: architectures \times properties -> risks (return the value of the objective function for the given property, that is, the protection level, computed on the given architecture)
def computeArchitectureRiskLevelForPAAndP(architecture, pointOfAttack, property):
    
    l = computeLikelihood(pointOfAttack)
    targets = set()

    if (not isinstance(pointOfAttack, frozenset)): # instead of "if (pointOfAttack in domains):" (otherwise, we should pass to this function the "domains" variable as well)
        architectureApex = [entityDomainPair for entityDomainPair in architecture if entityDomainPair[1] == pointOfAttack]
        for entityDomainPair in architectureApex:
            targets.add(entityDomainPair[0])
    else:
        architectureApex = [entityDomainPair for entityDomainPair in architecture if entityDomainPair[1] in pointOfAttack]
        for entityDomainPair1 in architectureApex:
            for entityDomainPair2 in architectureApex:
                if (entityDomainPair1[1] != entityDomainPair2[1]):
                    targets.add(frozenset([entityDomainPair1[0], entityDomainPair2[0]]))

    i = "None"
    for target in targets:
        j = computeTargetsAndImpact(target, property)
        if (j == "High"):
            i = j
        elif (j == "Medium" and i != "High"):
            i = j
        elif(j == "Low" and i != "High" and i != "Medium"):
            i = j

    return computeRiskLevel(i, l)

# function: return architectures with lowest risk levels
def adHoc(architecturesWithRiskLevels):
    
    architecturesWithMinimumRiskLevels = list()
    minimumRiskLevel = -1

    for architectureWithRiskLevels in architecturesWithRiskLevels:
        currentRiskLevel = computeRiskLevelAsWeightedNumber(architectureWithRiskLevels[0])

        if (minimumRiskLevel == -1 or currentRiskLevel < minimumRiskLevel):
            minimumRiskLevel = currentRiskLevel
            architecturesWithMinimumRiskLevels = list()
            architecturesWithMinimumRiskLevels.append(architectureWithRiskLevels)
        elif (currentRiskLevel == minimumRiskLevel):
            architecturesWithMinimumRiskLevels.append(architectureWithRiskLevels)

    return architecturesWithMinimumRiskLevels

# function: find Pareto Optimal architectures
def best(architecturesWithRiskLevels):

    architecturesParetoOptimalWithRiskLevels = list()
    discardedArchitecturesWithRiskLevels = list()

    while(len(architecturesParetoOptimalWithRiskLevels) + len(discardedArchitecturesWithRiskLevels) != len(architecturesWithRiskLevels)):

        # 1. Find a Pareto Optimal architecture
        remainingArchitecturesWithRiskLevels = list()
        for architectureWithRiskLevel in architecturesWithRiskLevels:
            if architectureWithRiskLevel not in discardedArchitecturesWithRiskLevels and architectureWithRiskLevel not in architecturesParetoOptimalWithRiskLevels:
                remainingArchitecturesWithRiskLevels.append(architectureWithRiskLevel)

        currentParetoOptimalArchitectureWithRiskLevel = random.sample(remainingArchitecturesWithRiskLevels , 1)[0]
        for remainingArchitectureWithRiskLevel in remainingArchitecturesWithRiskLevels:
            if (bestPareto(
                    remainingArchitectureWithRiskLevel,
                    currentParetoOptimalArchitectureWithRiskLevel
                )):
                currentParetoOptimalArchitectureWithRiskLevel = remainingArchitectureWithRiskLevel

        # 2. Remove architectures dominated by the found Pareto Optimal architecture
        architecturesParetoOptimalWithRiskLevels.append(currentParetoOptimalArchitectureWithRiskLevel)
        for remainingArchitectureWithRiskLevel in remainingArchitecturesWithRiskLevels:
            if (bestPareto(
                    currentParetoOptimalArchitectureWithRiskLevel,
                    remainingArchitectureWithRiskLevel
                )):
                discardedArchitecturesWithRiskLevels.append(remainingArchitectureWithRiskLevel)

    return architecturesParetoOptimalWithRiskLevels




# ==== helper functions ====

# helper function: return weighted sum of risk levels
def computeRiskLevelAsWeightedNumber(riskLevels):
    riskLevelAsNumber = 0
    for property, riskLevel in riskLevels.items():
        weight = computeWeight(property)
        if (riskLevel == "None"):
            riskLevelAsNumber = riskLevelAsNumber + 0
        elif (riskLevel == "Low"):
            riskLevelAsNumber = riskLevelAsNumber + 1 * weight
        elif (riskLevel == "Medium"):
            riskLevelAsNumber = riskLevelAsNumber + 2 * weight
        elif (riskLevel == "High"):
            riskLevelAsNumber = riskLevelAsNumber + 3 * weight
    return riskLevelAsNumber

# helper function: convert risk level to number
def computeRiskLevelAsNumber(riskLevel):
    riskValue = -1
    if (riskLevel == "None"):
        riskValue = 0
    elif (riskLevel == "Low"):
        riskValue = 1
    elif (riskLevel == "Medium"):
        riskValue = 2
    elif (riskLevel == "High"):
        riskValue = 3
    return riskValue

# helper function: check whether the first architecture is Pareto Optimal with respect to the second
def bestPareto(firstArchitectureWithRiskLevels, secondArchitectureWithRiskLevels):
    firstRiskLevels = firstArchitectureWithRiskLevels[0]
    secondRiskLevels = secondArchitectureWithRiskLevels[0]
    for objectiveFunction in firstRiskLevels:
        if (computeRiskLevelAsNumber(secondRiskLevels[objectiveFunction]) < computeRiskLevelAsNumber(firstRiskLevels[objectiveFunction])):
            return False
        
    for objectiveFunction in firstRiskLevels:
        if (computeRiskLevelAsNumber(firstRiskLevels[objectiveFunction]) < computeRiskLevelAsNumber(secondRiskLevels[objectiveFunction])):
            return True
        
    return False




# ==== functions added for FogAtlas ====

# rank architectures based on risk levels
def rankArchitectures(computeLikelihood, computeWeight, computeReplicas, prefilters, entities, domains):

    # base sets

    # resources that can be threathened by attackers
    #resources = set(["Data", "Encrypted Data", "Secret Keys", "Metadata"])

    # impacts of the risk assessment methodology
    #impacts = set(["None", "Low", "Medium", "High"])

    # likelihoods of the risk assessment methodology
    #likelihoods = set(["None", "Low", "Medium", "High"])

    # risk levels of the risk assessment methodology
    #risks = set(["None", "Low", "Medium", "High"])

    # properties refered to sensitive data, not to resources --- basically, our assets.
    properties = set(["C", "I", "A"])


    # derived sets

    # channels = {(d1, d2) \in domains \times domains | d1 \neq d2} (all possible communication channel between two domains)
    channels = set([frozenset([d1, d2]) for d1 in domains for d2 in domains if d1 < d2])
    
    # dataFlows =  {(e1, e2) \in entities \times entities | e1 \neq e2} (all possible data flows between two entities)
    #dataFlows = set([frozenset([e1, e2]) for e1 in entities for e2 in entities if e1 < e2])

    # pointsOfAttack = domains \cup channels (where an attacker might strike)
    pointsOfAttack = domains.union(channels)

    # targets = entities \cup dataFlows (possible targets of an attacker; a target is contained in a point of attack. In other words, a point of attack contains zero or more targets, while a target is contained in one or more points of attack)
    #targets = entities.union(dataFlows)

    # derived set (architectures \subseteq powerset(entities \times domains))
    architectures = powerset([(entity, domain) for entity in entities for domain in domains if (frozenset([entity, domain]) not in prefilters)])


    logging.debug("[TRADEOFFBOARD] There are " + str(len(architectures)) + " architectures")
    architecturesWithReplicas = set()
    for arc in architectures:
        isWithReplicas = True
        for entity in entities:
            replicasForEntity = len( [ pair[0] for pair in arc if (pair[0] == entity) ] )
            if ((replicasForEntity < (computeReplicas(entity))[0]) or (replicasForEntity > (computeReplicas(entity))[1])):
                isWithReplicas = False
        if (isWithReplicas):
            architecturesWithReplicas.add(arc)

    logging.debug("[TRADEOFFBOARD] With replicas, we consider " + str(len(architecturesWithReplicas)) + " architectures")
    architecturesWithRiskLevels = list()
    for arc in architecturesWithReplicas:
        riskLevels = {"C": "None", "I": "None", "A": "None"}
        for pointOfAttack in pointsOfAttack:
            for property in properties:
                tpm = computeArchitectureRiskLevelForPAAndP(arc, pointOfAttack, property)
                if (tpm == "High"):
                    riskLevels[property] = tpm
                elif (tpm == "Medium" and riskLevels[property] != "High"):
                    riskLevels[property] = tpm
                elif(tpm == "Low" and riskLevels[property] != "High" and riskLevels[property] != "Medium"):
                    riskLevels[property] = tpm

        architecturesWithRiskLevels.append( (riskLevels, arc) )

    if (weightsAreDefined):
        logging.debug("[TRADEOFFBOARD] Weights were defined, using the AdHoc algorithm")
    else:
        logging.debug("[TRADEOFFBOARD] Weights were not defined, using the Best algorithm")

    rankedArchitecturesWithRiskLevels = list()
    while (len(architecturesWithRiskLevels) != 0):
        if (weightsAreDefined):
            newArchitectures = adHoc(architecturesWithRiskLevels)
        else:
            newArchitectures = best(architecturesWithRiskLevels)
        rankedArchitecturesWithRiskLevels.extend(newArchitectures)
        for newArchitecture in newArchitectures:
            architecturesWithRiskLevels.remove(newArchitecture)

    rankedArchitectures = list()
    for riskLevel, arc in rankedArchitecturesWithRiskLevels:
        rankedArchitectures.append((riskLevel, arc))
    # for parc in rankedArchitectures:      
    #     print(str(parc))
    #     logging.debug("[TRADEOFFBOARD] ========")
        
    return rankedArchitectures

# wrapper for rankArchitectures function
def rankArchitectures_FogAtlas(entities, domains):

    rankedArchitecturesWithRiskLevels = rankArchitectures(computeLikelihood, computeWeight, computeReplicas, prefilters, entities, domains)
    rankedArchitectures = [pair[1] for pair in rankedArchitecturesWithRiskLevels]
    return rankedArchitectures




# Below, code for testing purposes

# the entities (i.e., microservices)
entities = set(["cryptoac-proxy-a", "cryptoac-proxy-b", "cryptoac-mm", "cryptoac-dm", "front-end-pub", "front-end-sub"])

# # the domains (i.e., regions)
domains = set(["public", "private", "edge", "vehicle1", "vehicle2"])

logging.basicConfig(level=logging.DEBUG)

# # each architecture is of the form, e.g., "(('cryptoac-dm', 'cloudregion'), ('cryptoac-proxy', 'onpremiseregion'), ('cryptoac-mm', 'cloudregion'), ('cryptoac-rm', 'cloudregion'))"
rankingOfArchitectures = rankArchitectures(computeLikelihood, computeWeight, computeReplicas, prefilters, entities, domains)

for architecture in rankingOfArchitectures:
   print(architecture)