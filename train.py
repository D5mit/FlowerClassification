import argparse
import my_models

parser = argparse.ArgumentParser()
parser.add_argument("data_dir",        default='flowers', help='Folder where data is stored')
parser.add_argument('--save_dir',      default='checkpoint.pth', help='Set directory to save checkpoints')
parser.add_argument('--arch',          default='vgg16', help='Model, such as vgg13')
parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning, default 0.001')
parser.add_argument('--hidden_units',  default=256, type=float, help='Number of hidden units')
parser.add_argument('--epochs',        default=5, type=float, help='Number of epochs')
parser.add_argument('--gpu',           default=False, help='Set to True if GPU should be used')

args = parser.parse_args()

# font colors
CRED = '\033[90m'
CEND = '\033[0m'

def main():

    print('')
    print('WELCOME TO SOME TRAINING')
    print(CRED + '  ARGS: data_dir: ' + args.data_dir +
          ', save_dir: ' + args.save_dir +
          ', arch: ' + args.arch +
          ', learning_rate: ' +  str(args.learning_rate) +
          ', hidden_units: ' + str(args.hidden_units) +
          ', epochs: ' + str(args.epochs),
          ', gpu: ' + args.gpu + CEND)

    if args.arch == 'vgg16' or args.arch == 'alexnet':
        my_models.create_and_train_model(args.data_dir,
                              args.save_dir,
                              args.arch,
                              args.learning_rate,
                              args.hidden_units,
                              args.epochs,
                              args.gpu)
    else:
        print('Please select a valid training model')

# Call to get_input_args function to run the program
if __name__ == "__main__":
    main()


