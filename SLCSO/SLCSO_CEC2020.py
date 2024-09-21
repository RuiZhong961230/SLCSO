import os
from opfunu.cec_based.cec2020 import *
from copy import deepcopy


PopSize = 200
DimSize = 100
LB = [-100] * DimSize
UB = [100] * DimSize
TrialRuns = 30
MaxFEs = 1000 * DimSize

Pop = np.zeros((PopSize, DimSize))
Velocity = np.zeros((PopSize, DimSize))
FitPop = np.zeros(PopSize)
curFEs = 0
FuncNum = 1
phi = 0.1


# initialize the M randomly
def Initialization(func):
    global Pop, Velocity, FitPop
    Velocity = np.zeros((PopSize, DimSize))
    for i in range(PopSize):
        for j in range(DimSize):
            Pop[i][j] = LB[j] + (UB[j] - LB[j]) * np.random.rand()
        FitPop[i] = func(Pop[i])


def Check(indi):
    global LB, UB
    for i in range(DimSize):
        range_width = UB[i] - LB[i]
        if indi[i] > UB[i]:
            n = int((indi[i] - UB[i]) / range_width)
            mirrorRange = (indi[i] - UB[i]) - (n * range_width)
            indi[i] = UB[i] - mirrorRange
        elif indi[i] < LB[i]:
            n = int((LB[i] - indi[i]) / range_width)
            mirrorRange = (LB[i] - indi[i]) - (n * range_width)
            indi[i] = LB[i] + mirrorRange
        else:
            pass
    return indi


def SLCSO(func):
    global Pop, Velocity, FitPop, phi, curFEs
    Off = np.zeros((PopSize, DimSize))
    FitOff = np.zeros(PopSize)
    Xbest = Pop[np.argmin(FitPop)]
    k = 5
    for i in range(PopSize):
        candi = list(range(PopSize))
        candi.remove(i)
        Competitors = np.random.choice(candi, k, replace=False)
        FitCompetitors = FitPop[Competitors]
        if FitPop[i] < np.median(FitCompetitors):
            Off[i] = deepcopy(Pop[i])
            FitOff[i] = FitPop[i]
        else:
            win_idx = Competitors[np.argmin(FitCompetitors)]
            Velocity[i] = np.random.rand(DimSize) * Velocity[i] + np.random.rand(DimSize) * (Pop[win_idx] - Pop[i]) + phi * (Xbest - Pop[i])
            Off[i] = Pop[i] + Velocity[i]
            Off[i] = Check(Off[i])
            FitOff[i] = func(Off[i])
            curFEs += 1
    FitPop = FitOff
    Pop = deepcopy(Off)


def RunSLCSO(func):
    global FitPop, curFEs, MaxFEs, TrialRuns, DimSize
    All_Trial_Best = []
    MAX = 0
    for i in range(TrialRuns):
        BestList = []
        curFEs = 0
        np.random.seed(2000 + 22 * i)
        Initialization(func)
        BestList.append(min(FitPop))
        while curFEs < MaxFEs:
            SLCSO(func)
            BestList.append(min(FitPop))
        MAX = max(len(BestList), MAX)
        All_Trial_Best.append(BestList)
    for i in range(len(All_Trial_Best)):
        for j in range(len(All_Trial_Best[i]), MAX):
            All_Trial_Best[i].append(All_Trial_Best[i][-1])
    np.savetxt("./SLCSO_Data/CEC2020/F" + str(FuncNum+1) + ".csv", All_Trial_Best, delimiter=",")


def main(dim):
    global FuncNum, DimSize, MaxFEs, MaxIter, Pop, LB, UB
    DimSize = dim
    Pop = np.zeros((PopSize, dim))
    MaxFEs = dim * 1000
    MaxIter = int(MaxFEs / PopSize * 2)
    LB = [-100] * dim
    UB = [100] * dim

    CEC2020 = [F12020(Dim), F22020(Dim), F32020(Dim), F42020(Dim), F52020(Dim),
               F62020(Dim), F72020(Dim), F82020(Dim), F92020(Dim), F102020(Dim)]
    for i in range(len(CEC2020)):
        FuncNum = i
        RunSLCSO(CEC2020[i].evaluate)


if __name__ == "__main__":
    if os.path.exists('./SLCSO_Data/CEC2020') == False:
        os.makedirs('./SLCSO_Data/CEC2020')
    Dims = [100]
    for Dim in Dims:
        main(Dim)


