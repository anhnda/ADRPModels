import openbabel
import numpy as np

obConversion = openbabel.OBConversion()
obConversion.SetInAndOutFormats("smi", "mdl")




def simpleGenECFP(smile):
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, smile)
    ecfp = []
    for atom in openbabel.OBMolAtomIter(mol):
        if atom.IsHydrogen():
            continue
        atomECFP = np.ndarray(8, dtype=float)
        atomECFP[0] = atom.GetHvyValence()
        atomECFP[1] = atom.BOSum()
        atomECFP[2] = atom.GetAtomicNum()
        atomECFP[3] = atom.GetIsotope()
        atomECFP[4] = atom.GetFormalCharge()
        atomECFP[5] = atom.ExplicitHydrogenCount() + atom.ImplicitHydrogenCount()
        atomECFP[6] = atom.IsInRing()
        atomECFP[7] = atom.IsAromatic()
        ecfp.append(atomECFP)
    ecfp = np.vstack(ecfp)
    return ecfp


def getAtomProperties(atom):
    atomProperties = [atom.GetFormalCharge(), atom.GetAtomicNum(), atom.GetIsotope(), atom.GetSpinMultiplicity(),
                      atom.GetAtomicMass(), atom.GetExactMass(), atom.GetValence(), atom.GetHyb(),
                      atom.GetImplicitValence(), atom.GetHvyValence(), atom.GetHeteroValence(), atom.IsHydrogen(),
                      atom.IsHydrogen(), atom.IsCarbon(), atom.IsNitrogen(), atom.IsOxygen(), atom.IsSulfur(),
                      atom.IsPhosphorus(), atom.IsAromatic(), atom.IsInRing(), atom.IsInRingSize(5),
                      atom.IsInRingSize(6), atom.IsInRingSize(7), atom.IsInRingSize(8), atom.IsInRingSize(9),
                      atom.IsHeteroatom(), atom.IsNotCorH(), atom.IsCarboxylOxygen(), atom.IsPhosphateOxygen(),
                      atom.IsSulfateOxygen(), atom.IsNitroOxygen(), atom.IsAmideNitrogen(), atom.IsPolarHydrogen(),
                      atom.IsNonPolarHydrogen(), atom.IsAromaticNOxide(), atom.IsChiral(), atom.IsAxial(),
                      atom.IsClockwise(), atom.IsAntiClockwise(), atom.IsPositiveStereo(), atom.IsNegativeStereo(),
                      atom.HasChiralitySpecified(), atom.HasChiralVolume(), atom.IsHbondAcceptor(), atom.IsHbondDonor(),
                      atom.IsHbondDonorH(), atom.IsMetal(), atom.HasNonSingleBond(), atom.HasSingleBond(),
                      atom.HasBondOfOrder(1), atom.HasBondOfOrder(2), atom.HasBondOfOrder(3), atom.HasAromaticBond()]

    size = len(atomProperties)
    ar = np.ndarray(size, dtype=float)
    for i in range(size):
        ar[i] = float(atomProperties[i])
    return ar


def fullPass1GenECFP(smile):
    mol = openbabel.OBMol()
    obConversion.ReadString(mol, smile)
    ecfp = []
    for atom in openbabel.OBMolAtomIter(mol):
        if atom.IsHydrogen():
            continue
        atomECFP = getAtomProperties(atom)
        ecfp.append(atomECFP)
    ecfp = np.vstack(ecfp)
    return ecfp


def genECFPLiuData():
    pass


if __name__ == "__main__":
    smile = "OCCN1CCN(CCCN2C3=CC=CC=C3SC3=C2C=C(Cl)C=C3)CC1"
    print(fullPass1GenECFP(smile))
