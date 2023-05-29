import roboflow

# Instantiate Roboflow object with your API key
rf = roboflow.Roboflow(api_key="WD7eEBRMv65lAzKonx9J")

# List all projects for your workspace
workspace = rf.workspace()
print(workspace)

# Load a certain project (workspace url is optional)
# project = rf.workspace("movses-movsesyan-pnofn").project("elec-stuff")
project = rf.project("elec-stuff")
version = project.version(18)

version.deploy("yolov8", "houseplan/elecStuffBigger17")

##New Version: pip install ultralytics==8.0.20
##Old Version: version=8.0.43