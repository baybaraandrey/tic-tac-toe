import random

import sys

from AI import QModel, FilePersistent

from game import Game



def usage():
    print('Usage: game [--load <path>|--save <path>]')


def run_game(model):
    game = Game(model)
    game.run()


if __name__ == '__main__':
    # uncomment this for reprodusing states
    # random.seed(13)

    random.seed()
    args = sys.argv[1:]
    if len(args) < 2:
        usage()
        exit(1)

    model = QModel()
    flag = args[0]
    if flag == '--load':
        model = FilePersistent.load_from(args[1])
    elif flag == '--save':
        model.fit()
        FilePersistent.save_to(model, args[1])
    else:
        usage()
        exit(1)


    try:
        run_game(model)
    except KeyboardInterrupt:
        print('Bye bye!')
