import re


def _str2tuple(string):
    """Convert shape string to list, internal use only.

    Parameters
    ----------
    string: str
        Shape string.

    Returns
    -------
    list of str
        Represents shape.
    """
    return re.findall(r"\d+", string)


def calConvParams(node, nodes, shape_dict, heads):
    """print layer information

    Parameters
    ----------
    node: dict
        Node information.
    out_shape: dict
        Node shape information.
    Returns
    ------
        Node total parameters.
    """

    op = node["op"]
    pre_node = []
    pre_filter = 0
    if op != "null":
        inputs = node["inputs"]
        for item in inputs:
            input_node = nodes[item[0]]
            input_name = input_node["name"]
            if input_node["op"] != "null" or item[0] in heads:
                # add precede
                pre_node.append(input_name)
                if input_node["op"] != "null":
                    key = input_name + "_output"
                else:
                    key = input_name
                if key in shape_dict:
                    shape = shape_dict[key][1:]
                    pre_filter = pre_filter + int(shape[0])
    cur_param = 0
    if op == 'Convolution':
        if "no_bias" in node["attrs"] and node["attrs"]["no_bias"] == 'True':
            num_group = int(node['attrs'].get('num_group', '1'))
            cur_param = pre_filter * int(node["attrs"]["num_filter"]) \
                        // num_group
            for k in _str2tuple(node["attrs"]["kernel"]):
                cur_param *= int(k)
        else:
            num_group = int(node['attrs'].get('num_group', '1'))
            cur_param = pre_filter * int(node["attrs"]["num_filter"]) \
                        // num_group
            for k in _str2tuple(node["attrs"]["kernel"]):
                cur_param *= int(k)
            cur_param += int(node["attrs"]["num_filter"])
    elif op == 'FullyConnected':
        if "no_bias" in node["attrs"] and node["attrs"]["no_bias"] == 'True':
            cur_param = pre_filter * int(node["attrs"]["num_hidden"])
        else:
            cur_param = (pre_filter + 1) * int(node["attrs"]["num_hidden"])
    elif op == 'BatchNorm':
        key = node["name"] + "_output"
        num_filter = shape_dict[key][1]
        cur_param = int(num_filter) * 2
    elif op == 'Embedding':
        cur_param = int(node["attrs"]['input_dim']) * int(node["attrs"]['output_dim'])
    return cur_param
