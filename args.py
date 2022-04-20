import argparse
import os
parser = argparse.ArgumentParser()


parser.add_argument("--tiny_sample", dest="tiny_sample", action="store_true") 
parser.add_argument("--num_prototypes", type=int, default=20)
parser.add_argument("--num_pos_prototypes", type=int, default=19)
parser.add_argument("--modelname", type=str, default=None)
parser.add_argument("--cuda", type=str, default=None)
parser.add_argument("--rec_loss", type=int, default=0)
parser.add_argument("--train_bal", type=int, default=0)
parser.add_argument("--train_decoder", type=int, default=0)



parser.add_argument("--model", type=str, default="PrototExNL")



args = parser.parse_args()
