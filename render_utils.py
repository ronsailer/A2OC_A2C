#!/usr/bin/python

import sys
import retro
from os import listdir
from os.path import isfile, join, isdir


def render(file):
    movie = retro.Movie(file)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    env.reset()
    frame = 0
    framerate = 2
    while movie.step():
        if frame == framerate:
            env.render()
            frame = 0
        else:
            frame += 1

        keys = []
        for i in range(env.NUM_BUTTONS):
            keys.append(movie.get_key(i))
        _obs, _rew, _done, _info = env.step(keys)
    env.close()
if isdir(sys.argv[1]):
    onlyfiles = [f for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
    onlyfiles.sort()
    for file in onlyfiles:
        if ".bk2" in file :
            print('playing', file)
            render(sys.argv[1]+file)
else:
    print('playing', sys.argv[1])
    render(sys.argv[1])
