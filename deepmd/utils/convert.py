import os
import json
import logging
from deepmd.env import tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from typing import Optional

from deepmd.common import j_loader
from deepmd.env import tf, GLOBAL_ENER_FLOAT_PRECISION
from deepmd.utils.argcheck import normalize
from deepmd.utils.compat import update_deepmd_input
from deepmd.utils.errors import GraphTooLargeError, GraphWithoutTensorError
from deepmd.utils.graph import get_tensor_by_name
from deepmd.entrypoints.freeze import freeze
from deepmd.entrypoints.train import train, get_rcut, get_min_nbor_dist
from deepmd.entrypoints.transfer import transfer

log = logging.getLogger(__name__)


def convert_13_to_21(input_model: str, output_model: str):
    """Convert DP 1.3 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_13_to_21(input_model: str, output_model: str):
    """Convert DP 1.3 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_12_to_21(input_model: str, output_model: str):
    """Convert DP 1.2 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp12_to_dp13('frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_10_to_21(input_model: str, output_model: str):
    """Convert DP 1.0 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp10_to_dp11('frozen_model.pbtxt')
    convert_dp12_to_dp13('frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_012_to_21(input_model: str, output_model: str):
    """Convert DP 0.12 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp012_to_dp10('frozen_model.pbtxt')
    convert_dp10_to_dp11('frozen_model.pbtxt')
    convert_dp12_to_dp13('frozen_model.pbtxt')
    convert_dp13_to_dp20('frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)


def convert_20_to_21(input_model: str, output_model: str):
    """Convert DP 2.0 graph to 2.1 graph.
    
    Parameters
    ----------
    input_model : str
        filename of the input graph
    output_model : str
        filename of the output graph
    """
    convert_pb_to_pbtxt(input_model, 'frozen_model.pbtxt')
    convert_dp20_to_dp21('frozen_model.pbtxt')
    convert_pbtxt_to_pb('frozen_model.pbtxt', output_model)
    if os.path.isfile('frozen_model.pbtxt'):
        os.remove('frozen_model.pbtxt')
    print("the converted output model (2.1 support) is saved in %s" % output_model)

def convert_pb_to_pbtxt(pbfile: str, pbtxtfile: str):
    """Convert DP graph to graph text.
    
    Parameters
    ----------
    pbfile : str
        filename of the input graph
    pbtxtfile : str
        filename of the output graph text
    """
    with gfile.FastGFile(pbfile, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tf.train.write_graph(graph_def, './', pbtxtfile, as_text=True)

def convert_pbtxt_to_pb(pbtxtfile: str, pbfile: str):
    """Convert DP graph text to graph.
    
    Parameters
    ----------
    pbtxtfile : str
        filename of the input graph text
    pbfile : str
        filename of the output graph
    """
    with tf.gfile.FastGFile(pbtxtfile, 'r') as f:
        graph_def = tf.GraphDef()
        file_content = f.read()
        # Merges the human-readable string in `file_content` into `graph_def`.
        text_format.Merge(file_content, graph_def)
        tf.train.write_graph(graph_def, './', pbfile, as_text=False)


def convert_dp012_to_dp10(file: str):
    """Convert DP 1.0 graph text to 1.1 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(file) as fp:
        file_content = fp.read()
    file_content = file_content\
                   .replace('DescrptNorot', 'DescrptSeA') \
                   .replace('ProdForceNorot', 'ProdForceSeA') \
                   .replace('ProdVirialNorot', 'ProdVirialSeA')
    with open(file, 'w') as fp:
        fp.write(file_content)


def convert_dp10_to_dp11(file: str):
    """Convert DP 1.0 graph text to 1.1 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(file, 'a') as f:
        f.write("""
node {
  name: "fitting_attr/daparam"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }                                                                                                                                                 }
}
""")


def convert_dp12_to_dp13(file: str):
    """Convert DP 1.2 graph text to 1.3 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        ii = 0
        lines = f.readlines()
        while (ii < len(lines)):
            line = lines[ii]
            file_data += line
            ii+=1
            if 'name' in line and ('DescrptSeA' in line or 'ProdForceSeA' in line or 'ProdVirialSeA' in line):
                while not('attr' in lines[ii] and '{' in lines[ii]):
                    file_data += lines[ii]
                    ii+=1
                file_data += '  attr {\n'
                file_data += '    key: \"T\"\n'
                file_data += '    value {\n'
                file_data += '      type: DT_DOUBLE\n'
                file_data += '    }\n'
                file_data += '  }\n'
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)


def convert_dp13_to_dp20(fname: str):
    """Convert DP 1.3 graph text to 2.0 graph text.
    
    Parameters
    ----------
    file : str
        filename of the graph text
    """
    with open(fname) as fp:
        file_content = fp.read()
    file_content += """
node {
  name: "model_attr/model_version"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "1.0"
      }
    }
  }
}
"""
    file_content = file_content\
                   .replace('DescrptSeA', 'ProdEnvMatA')\
                   .replace('DescrptSeR', 'ProdEnvMatR')
    with open(fname, 'w') as fp:
        fp.write(file_content)

def convert_dp20_to_dp21(fname: str):
    with open(fname) as fp:
        file_content = fp.read()
    old_model_version_node = """
node {
  name: "model_attr/model_version"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "1.0"
      }
    }
  }
}
"""
    new_model_version_node = """
node {
  name: "model_attr/model_version"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_STRING
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_STRING
        tensor_shape {
        }
        string_val: "1.1"
      }
    }
  }
}
"""
    file_content = file_content\
                   .replace(old_model_version_node, new_model_version_node)\
                   .replace('TabulateFusion', 'TabulateFusionSeA')\
                   .replace('TabulateFusionGrad', 'TabulateFusionSeAGrad')\
                   .replace('TabulateFusionGradGrad', 'TabulateFusionSeAGradGrad')
    with open(fname, 'w') as fp:
        fp.write(file_content)

def convert_org_to_ascend(
    input: str,
    output: str,
    checkpoint_folder: str,
    training_script: str,
    mpi_log: str,
    log_path: Optional[str],
    log_level: int,
    **kwargs
):
    """convert trained model to Ascend mix-precision model.

    Generate a Ascend_transfer mix-precision model.

    Parameters
    ----------
    input : str
        frozen model file to compress
    output : str
        compressed model filename
    checkpoint_folder : str
        trining checkpoint folder for freezing
    training_script : str
        training script of the input frozen model
    mpi_log : str
        mpi logging mode for training
    log_path : Optional[str]
        if speccified log will be written to this file
    log_level : int
        logging level
    """
    try:
        t_jdata = get_tensor_by_name(input, 'train_attr/training_script')
        t_min_nbor_dist = get_tensor_by_name(input, 'train_attr/min_nbor_dist')
        jdata = json.loads(t_jdata)
    except GraphWithoutTensorError as e:
        if training_script == None:
            raise RuntimeError(
                "The input frozen model: %s has no training script or min_nbor_dist information, "
                "which is not supported by the model compression interface. "
                "Please consider using the --training-script command within the model compression interface to provide the training script of the input frozen model. "
                "Note that the input training script must contain the correct path to the training data." % input
            ) from e
        elif not os.path.exists(training_script):
            raise RuntimeError(
                "The input training script %s (%s) does not exist! Please check the path of the training script. " % (input, os.path.abspath(input))
            ) from e
        else:
            log.info("stage 0: compute the min_nbor_dist")
            jdata = j_loader(training_script)
            jdata = update_deepmd_input(jdata)
            t_min_nbor_dist = get_min_nbor_dist(jdata, get_rcut(jdata))

    _check_transfer_model_type(input)

    tf.constant(t_min_nbor_dist,
        name = 'train_attr/min_nbor_dist',
        dtype = GLOBAL_ENER_FLOAT_PRECISION)
    jdata["model"]["descriptor"]["precision"] = "float16"
    jdata["model"]["fitting_net"]["precision"] = "float16"
    jdata["model"]["transfered_from_model"] = True
    jdata["training"]["save_ckpt"] = os.path.join("model-transfer", "model.ckpt")
    jdata = update_deepmd_input(jdata)
    jdata = normalize(jdata)

    # check the descriptor info of the input file
    # move to the specific Descriptor class

    # stage 1: training or refining the model with tabulation
    log.info("\n\n")
    log.info("stage 1: generate the mix-precision model")
    control_file = "ascend-transfer.json"
    with open(control_file, "w") as fp:
        json.dump(jdata, fp, indent=4)
    try:
        train(
            INPUT=control_file,
            init_model=None,
            restart=None,
            init_frz_model=input,
            output=control_file,
            mpi_log=mpi_log,
            log_level=log_level,
            log_path=log_path,
            is_compress=False,
            skip_neighbor_stat=True,
        )
    except GraphTooLargeError as e:
        raise RuntimeError(
            "The uniform step size of the tabulation's first table is %f, " 
            "which is too small. This leads to a very large graph size, "
            "exceeding protobuf's limitation (2 GB). You should try to "
            "increase the step size." % step
        ) from e

    # reset the graph, otherwise the size limitation will be only 2 GB / 2 = 1 GB
    tf.reset_default_graph()

    # stage 2: freeze the mix-precision model
    log.info("\n\n")
    log.info("stage 2: freeze the mix-precision model")
    try:
        freeze(checkpoint_folder=checkpoint_folder, output=output, node_names=None)
    except GraphTooLargeError as e:
        raise RuntimeError(
            "The uniform step size of the tabulation's first table is %f, " 
            "which is too small. This leads to a very large graph size, "
            "exceeding protobuf's limitation (2 GB). You should try to "
            "increase the step size." % step
        ) from e

    # stage 3: transfer the mix-precision model
    log.info("\n\n")
    log.info("stage 3: transfer the mix-precision model")
    transfer(old_model=input, raw_model=output, output=output)

def _check_transfer_model_type(model_file):
    try:
        t_model_type = bytes.decode(get_tensor_by_name(model_file, 'model_type'))
    except GraphWithoutTensorError as e:
        # Compatible with the upgraded model, which has no 'model_type' info
        t_model_type = None
    
    if t_model_type == "ascend_transfer_model":
        raise RuntimeError("The input model %s has already been compressed! Please do not compress the model repeatedly. " % model_file)