import random
"""
for _ in number_of_testcases:
  xs = [4_random_punkte_die_paarweise_verschieden_sind]
  xs.sort()
  p1,p2 = (xs[0], xs[1]), (xs[2], xs[3])
"""


def main():
    for _ in range(20):
        xs = [random.sample(range(-20, 20), 4)]
        xs.sort()
        p1, p2 = (xs[0], xs[1]), (xs[2], xs[3])
        print(f"{p1}, {p2}")


if __name__ == '__main__':
    main()