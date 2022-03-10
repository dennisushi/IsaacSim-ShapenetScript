# Custom functions for IsaacSim's ShapeNet support by Dennis Hadjivelichkov

from omni.isaac.shapenet import utils
shapenet_name_to_id_dict = utils.LABEL_TO_SYNSET
from omni.isaac.shapenet.globals import *
import glob

import omni.client
import omni.kit
import omni.usd

import asyncio
import os
from pxr import UsdGeom, Gf, Tf

import requests
import urllib.request
import shutil
import sys

from omni.isaac.shapenet.shape import *
from pxr import Gf

def get_dict():
  return shapenet_name_to_id_dict

def get_id(s):
  if s in list(shapenet_name_to_id_dict.keys()):
    return shapenet_name_to_id_dict[s]
  return None

def get_random_id():
  import random
  keys = list(shapenet_name_to_id_dict.keys())
  i = random.randint(0, len(keys))
  s = keys[i]
  return shapenet_name_to_id_dict[s]

def get_random_obj(id):
  loc = get_local_shape_loc()
  subdirs = glob.glob(loc+"/%s/*"%id)
  import random 
  subdir = random.choice(subdirs)
  return subdir.split("/")[-1]

def get_random_existing_id():
  """
  Returns the id of a random shapenet class that already exists locally
  """
  loc = get_local_shape_loc()
  subdirs = [f for f in glob.glob(loc+"/*") if f.split("/")[-1][0]=="0"]
  import random 
  subdir = random.choice(subdirs)
  return subdir.split("/")[-1]

def get_random_existing_obj(id):
  """
  Returns the id of a random shapenet object belonging to given class (string: id)
  that already exists locally
  """
  loc = get_local_shape_loc()
  subdirs = glob.glob(loc+"/%s/*"%id)
  import random 
  subdir = random.choice(subdirs)
  return subdir.split("/")[-1]

# The following is a modified version of addShapePrim - only difference is that it takes
# a shape name and is asynchronous

# This is the main entry point for any function that wants to add a shape to the scene.
# Care must be taken when running this on a seperate thread from the main thread because
# it calls c++ modules from python which hold the GIL.
async def addShapePrim_sync(
    omniverseServer, synsetId, modelId, pos, rot, scale, 
    auto_add_physics, use_convex_decomp, do_not_place=False,
    shape_name = None):
    # use shapenet v2 for models
    shape_url = g_shapenet_url + "2/" + synsetId + "/" + modelId + "/"
    # Get the local file system path and the omni server path
    local_folder = get_local_shape_loc() + "/" + synsetId + "/" + modelId + "/"
    local_path = local_folder + "models/model_normalized.obj"
    local_modified_path = local_folder + "models/modified/model.obj"

    global g_omni_shape_loc
    omni_shape_loc = "omniverse://" + omniverseServer + g_omni_shape_loc

    (result, entry) = await omni.client.stat_async(omni_shape_loc)
    if not result == omni.client.Result.OK:
        print("Saving converted files locally since omniverse server is not connected")
        omni_shape_loc = get_local_shape_loc() + "/local-converted-USDs"

    omni_path = (
        omni_shape_loc + "/n" + synsetId + "/i" + modelId + "/"
    )  # don't forget to add the name at the end and .usd
    omni_modified_path = omni_shape_loc + "/n" + synsetId + "/i" + modelId + "/modified/"

    stage = omni.usd.get_context().get_stage()
    if not stage:
        return "ERROR Could not get the stage."

    # Get the name of the shapenet object reference in the stage if it exists
    # (i.e. it has been added already and is used in another location on the stage).
    synsetID_path = g_root_usd_namespace_path + "/n" + synsetId
    over_path = synsetID_path + "/i" + modelId

    # Get the name of the instance we will add with the transform, this is the actual visible prim
    # instance of the reference to the omniverse file which was converted to local disk after
    global g_shapenet_db
    g_shapenet_db = get_database()
    if shape_name is None:
      if g_shapenet_db == None:
          shape_name = ""
          print("Please create an Shapenet ID Database with the menu.")
      else:
          shape_name = Tf.MakeValidIdentifier(g_shapenet_db[synsetId][modelId][4])
      if shape_name == "":
          shape_name = "empty_shape_name"
    
    prim_path = str(stage.GetDefaultPrim().GetPath()) + "/" + shape_name
    prim_path_len = len(prim_path)
    shape_name_len = len(shape_name)

    # if there is only one instance, we don't add a _# postfix, but if there is multiple, then the second instance
    # starts with a _1 postfix, and further additions increase that number.
    insta_count = 0
    while stage.GetPrimAtPath(prim_path):
        insta_count += 1
        prim_path = f"{prim_path[:prim_path_len]}_{insta_count}"
        shape_name = f"{shape_name[:shape_name_len]}_{insta_count}"

    omni_path = omni_path + shape_name + ".usd"
    omni_modified_path = omni_modified_path + shape_name + ".usd"
    # If the prim refernce to the omnivers file is not already on
    # the stage the stage we will need to add it.
    place_later = False
    if not stage.GetPrimAtPath(over_path):
        print(f"-Shapenet is adding {shape_name} to the stage for the first time.")
        # If the files does not already exist in omniverse we will have to add it there
        # with our automatic conversion of the original shapenet file.
        # We need to check if the modified file is on disk, so if it's not on the omni server it will
        # be added there even if the non modified one already exists on omni.
        if os.path.exists(local_modified_path) or file_exists_on_omni(omni_modified_path):
            omni_path = omni_modified_path
        if not file_exists_on_omni(omni_path):
            # If the original omniverse file does not exist locally, we will have to pull
            # it from Stanford's shapenet database on the web.
            if os.path.exists(local_modified_path):
                local_path = local_modified_path
                omni_path = omni_modified_path
            if not os.path.exists(local_path):
                # Pull the shapenet files to the local drive for conversion to omni:usd
                print(f"--Downloading {local_folder} from {shape_url}.")
                download_folder(local_folder, shape_url)
            # Add The file to omniverse here, if you add them asyncronously, then you have to do the
            # rest of the scene adding later.
            print(f"---Converting {shape_name}...")
            status = await convert(local_path, omni_path)
            if not status:
                return f"ERROR OmniConverterStatus is {status}"
            print(f"---Added to Omniverse as {omni_path}.")

        # Add the over reference of the omni file to the stage here.
        print(f"----Adding over of {over_path} to stage.")
        if not do_not_place and not place_later:
            over = stage.OverridePrim(over_path)
            over.GetReferences().AddReference(omni_path)

    # Add the instance of the shape here.
    if not do_not_place and not place_later:
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddInternalReference(over_path)

        metersPerUnit = UsdGeom.GetStageMetersPerUnit(stage)
        scaled_scale = scale / metersPerUnit
        addobject_fn(prim.GetPath(), pos, rot, scaled_scale)

        # add physics
        if auto_add_physics:
            from omni.physx.scripts import utils

            print("Adding PHYSICS to ShapeNet model")
            shape_approximation = "convexHull"
            if use_convex_decomp:
                shape_approximation = "convexDecomposition"
            utils.setRigidBody(prim, shape_approximation, False)

        return prim

    return None

async def add_random_obj_sync(
    omniverseServer= 'localhost', synsetId=None, modelId=None, pos=Gf.Vec3d(0.,0.,0.), rot=Gf.Rotation((0.0,0.0,0.0), 0.0), scale=1.0, 
    auto_add_physics=True, use_convex_decomp=True, do_not_place=False,
    shape_name = None):
  
  if synsetId is None:
    synsetId = get_random_existing_id()
  if modelId is None:
    modelId = get_random_existing_obj(synsetId)

  return await addShapePrim_sync(
                  omniverseServer, synsetId, modelId, pos, rot, scale, 
                  auto_add_physics, use_convex_decomp, do_not_place,
                  shape_name)
