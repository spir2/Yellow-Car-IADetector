import os
import sys
#-------------------------------Run_a_PATH_in_VSCODE ------------------------------
if sys.argv:
    filepath = sys.argv[0]
    folder, filename = os.path.split(filepath)
    os.chdir(folder) # now your working dir is the parent folder of the script
#----------------------------------------------------------------------------------
