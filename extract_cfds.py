from packages import iFeatureOmegaCLI

# Snippet for extracting BLOSUM62

protein = iFeatureOmegaCLI.iProtein("data/Independent.txt")
protein.get_descriptor("BLOSUM62")
print(protein.encodings)
protein.to_csv("features/CFDs/BLOSUM62_Ind.csv", "index=False", header=True)