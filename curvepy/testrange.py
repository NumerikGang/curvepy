
x = list(itertools.chain(*([(A,B), (B,C), (A,C)] for A,B,C in [((p, neighbour), (p, fst_collision), (neighbour, fst_collision)),
                                ((p, neighbour), (p, snd_collision), (neighbour, snd_collision))])))