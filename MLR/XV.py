import numpy as np



def XV(YObsTrain, yObsTest, yPredTest):
    # Initialize variables
    Q2F1 = 0.0
    Q2F2 = 0.0
    scaleAvgRm2Test = 0.0
    scaleDeltaRm2Test = 0.0
    CCCTest = 0.0
    MAE100Test = 0.0
    SD100Test = 0.0
    rangeTrain = 0.0
    Q2F195 = 0.0
    Q2F295 = 0.0
    scaleAvgRm2Test95 = 0.0
    scaleDeltaRm2Test95 = 0.0
    CCCTest95 = 0.0
    SD95Test = 0.0
    MAE95Test = 0.0

    meanYObsTrain = np.mean(YObsTrain)
    nCompT = len(yObsTest)
    nComp = len(YObsTrain)

    yObsT = np.array(yObsTest)
    yPredT = np.array(yPredTest)
    yPredTCopy = np.array(yPredTest)

    meanYObsTest = np.mean(yObsT)
    maxYObsT = np.max(yObsT)
    minYObsT = np.min(yObsT)

    scaleYObsT = (yObsT - minYObsT) / (maxYObsT - minYObsT)
    scaleYPredT = (yPredT - minYObsT) / (maxYObsT - minYObsT)

    residual2T = np.zeros(nCompT)

    sumYObsT = 0.0
    scaleSumYObsT = 0.0
    sumYPredT = 0.0
    scaleSumYPredT = 0.0
    yObsYPredT = 0.0
    scaleYObsYPredT = 0.0
    yPred2T = 0.0
    scaleYPred2T = 0.0
    yObs2T = 0.0
    scaleYObs2T = 0.0
    r2PredNT = 0.0
    scaleR2PredNT = 0.0
    r2PredDT = 0.0

    for k in range(nCompT):
        sumYObsT += yObsT[k]
        scaleSumYObsT += scaleYObsT[k]
        sumYPredT += yPredT[k]
        scaleSumYPredT += scaleYPredT[k]
        yObsYPredT += yObsT[k] * yPredT[k]
        scaleYObsYPredT += scaleYObsT[k] * scaleYPredT[k]
        yPred2T += yPredT[k] * yPredT[k]
        scaleYPred2T += scaleYPredT[k] * scaleYPredT[k]
        yObs2T += yObsT[k] * yObsT[k]
        scaleYObs2T += scaleYObsT[k] * scaleYObsT[k]
        residual2T[k] = (yObsT[k] - yPredT[k]) ** 2
        r2PredNT += (yObsT[k] - yPredT[k]) ** 2
        scaleR2PredNT += (scaleYObsT[k] - scaleYPredT[k]) ** 2
        r2PredDT += (yObsT[k] - meanYObsTrain) ** 2

    scaleAvgYObsT = scaleSumYObsT / nCompT
    avgYPredT = sumYPredT / nCompT
    scaleAvgYPredT = scaleSumYPredT / nCompT
    kT = yObsYPredT / yPred2T
    scaleKT = scaleYObsYPredT / scaleYPred2T
    kDashT = yObsYPredT / yObs2T
    scaleKDashT = scaleYObsYPredT / scaleYObs2T

    Q2F1 = 1.0 - r2PredNT / r2PredDT
    RMSEPT = np.sqrt(r2PredNT / nCompT)

    RYPredYObsT = 0.0
    scaleRYPredYObsT = 0.0
    yPredYBar2T = 0.0
    scaleYPredYBar2T = 0.0
    yObsYBar2T = 0.0
    scaleYObsYBar2T = 0.0
    yKT = 0.0
    scaleYKT = 0.0
    yKDashT = 0.0
    scaleYKDashT = 0.0

    for m in range(nCompT):
        RYPredYObsT += (yPredT[m] - avgYPredT) * (yObsT[m] - meanYObsTest)
        scaleRYPredYObsT += (scaleYPredT[m] - scaleAvgYPredT) * (scaleYObsT[m] - scaleAvgYObsT)
        yPredYBar2T += (yPredT[m] - avgYPredT) ** 2
        scaleYPredYBar2T += (scaleYPredT[m] - scaleAvgYPredT) ** 2
        yObsYBar2T += (yObsT[m] - meanYObsTest) ** 2
        scaleYObsYBar2T += (scaleYObsT[m] - scaleAvgYObsT) ** 2
        yKT += (yObsT[m] - kT * yPredT[m]) ** 2
        scaleYKT += (scaleYObsT[m] - scaleKT * scaleYPredT[m]) ** 2
        yKDashT += (yPredT[m] - kDashT * yObsT[m]) ** 2
        scaleYKDashT += (scaleYPredT[m] - scaleKDashT * scaleYObsT[m]) ** 2

    r2T = (RYPredYObsT / np.sqrt(yPredYBar2T * yObsYBar2T)) ** 2
    scaleR2 = (scaleRYPredYObsT / np.sqrt(scaleYPredYBar2T * scaleYObsYBar2T)) ** 2
    ro2 = 1.0 - yKT / yObsYBar2T
    scaleRo2 = 1.0 - scaleYKT / scaleYObsYBar2T
    ro2Dash = 1.0 - yKDashT / yPredYBar2T
    scaleRo2Dash = 1.0 - scaleYKDashT / scaleYPredYBar2T

    rm2 = r2T * (1.0 - np.sqrt(np.abs(r2T - ro2)))
    scaleRm2 = scaleR2 * (1.0 - np.sqrt(np.abs(scaleR2 - scaleRo2)))
    rm2Dash = r2T * (1.0 - np.sqrt(np.abs(r2T - ro2Dash)))
    scaleRm2Dash = scaleR2 * (1.0 - np.sqrt(np.abs(scaleR2 - scaleRo2Dash)))

    avgRm2 = (rm2 + rm2Dash) / 2.0
    scaleAvgRm2Test = (scaleRm2 + scaleRm2Dash) / 2.0

    deltaRm2 = max(rm2 - rm2Dash, rm2Dash - rm2)
    scaleDeltaRm2Test = max(scaleRm2 - scaleRm2Dash, scaleRm2Dash - scaleRm2)

    Q2F2 = 1.0 - r2PredNT / yObsYBar2T
    CCCTest = 2.0 * RYPredYObsT / (yPredYBar2T + yObsYBar2T + nCompT * (avgYPredT - meanYObsTest) ** 2)

    sumSD100 = 0.0
    sumForMAE = 0.0
    absResidualT = np.zeros(nCompT)
    residualSquareT = np.zeros(nCompT)

    for n in range(nCompT):
        residualSquareT[n] = np.abs(yObsT[n] - yPredT[n]) ** 2
        absResidualT[n] = np.abs(yObsT[n] - yPredT[n])
        sumForMAE += absResidualT[n]

    MAE100Test = sumForMAE / nCompT

    for n in range(nCompT):
        sumSD100 += (np.abs(yObsT[n] - yPredT[n]) - MAE100Test) ** 2

    SD100Test = np.sqrt(sumSD100 / (nCompT - 1))

    # Sorting based on residuals
    residual2T_sorted = np.sort(residual2T)

    nCompT5 = int(np.ceil(0.05 * nCompT))
    nCompT95 = nCompT - nCompT5

    SD95Test = 0.0
    for n in range(nCompT95, nCompT):
        SD95Test += residual2T_sorted[n]

    SD95Test = np.sqrt(SD95Test / (nCompT95))

    MAE95Test = np.sum(residual2T_sorted[:nCompT95]) / nCompT95

    test_error_sum = np.sum((yObsTest - yPredTest) ** 2)
    n_test = len(yObsTest)
    train_variance_sum = np.sum((YObsTrain - meanYObsTrain) ** 2)
    n_train = len(YObsTrain)
    Q2F3 = 1 - (test_error_sum / n_test) / (train_variance_sum / n_train)
    return {
        'Q2F1': Q2F1,
        'Q2F2': Q2F2,
        'Q2F3': Q2F3,
        'scaleAvgRm2Test': scaleAvgRm2Test,
        'scaleDeltaRm2Test': scaleDeltaRm2Test,
        'CCCTest': CCCTest,
        'MAE100Test': MAE100Test,
        'SD100Test': SD100Test,
        'rangeTrain': rangeTrain,
        'Q2F195': Q2F195,
        'Q2F295': Q2F295,
        'scaleAvgRm2Test95': scaleAvgRm2Test95,
        'scaleDeltaRm2Test95': scaleDeltaRm2Test95,
        'CCCTest95': CCCTest95,
        'SD95Test': SD95Test,
        'MAE95Test': MAE95Test
    }