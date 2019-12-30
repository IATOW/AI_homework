import torch
import os

from Net import Net


def run():
    example = torch.rand(1, 9, 17, 17)

    for maindir, subdir, file_name_list in os.walk("model"):
        for file in file_name_list:
            ext = file.split(".")[-1]
            if ext == "pt":
                d = int(file.split("_")[1])
                w = int(file.split("_")[2])

                model_black = Net(d, w, 9)
                model_black.load_state_dict(torch.load("model/" + file))
                model_black.eval()
                traced = torch.jit.trace(model_black, example)
                print(traced.code)
                traced.save("model/" + file + "s")




if __name__ == "__main__":
    run()