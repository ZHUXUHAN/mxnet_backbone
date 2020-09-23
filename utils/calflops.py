
def calConvFlops(inshape ,outshape ,kernel_size ,hasbias = True, num_group = 1, type='normal'):
    # inshape is [chanel , width , height]
    # kernel_size in [kernelwidth , kernelheight]
    # outshape is [chanel * width * height]
    # type = 'normal' 'depthwise' 'deformable'
    # return [multipflops,addflops,compare flops,expflops]
    addflops =0
    print(inshape, outshape, kernel_size)
    multipflops = inshape[0] * outshape[0] * outshape[1] * \
                  outshape[2] * kernel_size[0] * kernel_size[1] / num_group
    if hasbias:
        addflops = outshape[0] * outshape[1] * outshape[2]

    return [multipflops, addflops, 0, 0]


def calActivationFlops(inshape, outshape, type='relu'):
    # inshape is [chanel , width , height]
    # outshape is [chanel * width * height]
    # type = 'relu' 'sigmoid' 
    # return [multipflops,addflops,compare flops,expflops]
    compareflops = 1
    if type == 'relu':
        for shape in outshape:
            compareflops *= shape

    return [0, 0, compareflops, 0]


def calPoolingFlops(inshape, outshape, kernel_size, type='max'):
    # inshape is [chanel , width , height]
    # kernel_size in [kernelwidth , kernelheight]
    # outshape is [chanel * width * height]
    # type = 'max' 'ave' 'gop' 'deformable'
    # return [multipflops,addflops,compare flops,expflops]

    compareflops = 0
    addflops = 0
    multipflops = 0

    if type == 'max':
        compareflops += outshape[0] * outshape[1] * outshape[2] * kernel_size[0] * kernel_size[1]

    if type == 'ave':
        multipflops += outshape[0] * outshape[1] * outshape[2]
        addflops += outshape[0] * outshape[1] * outshape[2] * kernel_size[0] * kernel_size[1]

    if type == 'gop':
        multipflops += outshape[0] * outshape[1] * outshape[2]
        addflops += outshape[0] * outshape[1] * outshape[2] * inshape[0] * inshape[1]

    return [multipflops, addflops, compareflops, 0]


def calFcFlops(inshape, outshape, hasbias=True):
    # inshape is [Inchanel]
    # outshape is [Outchanel]
    # type = 'max' 'ave' 'gop'
    # return [multipflops,addflops,compare flops,expflops]

    addflops = 0
    multipflops = inshape[0] * outshape[0]

    if hasbias:
        addflops += outshape[0]

    return [multipflops, addflops, 0, 0]
