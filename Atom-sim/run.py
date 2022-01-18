import numpy as np
import matplotlib.pyplot as plt


def generate_points_regular(n, shape, min_dist):
    # compute grid shape based on number of points
    # width_ratio = shape[1] / shape[0]
    # num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    # num_x = np.int32(n / num_y) + 1

    num_x = np.int32(np.sqrt(n))
    num_y = np.int32(np.sqrt(n))

    # create regularly spaced neurons
    x = np.linspace(0.+min_dist/2, shape[1]-min_dist/2, num_x, dtype=np.float32)
    y = np.linspace(0.+min_dist/2, shape[0]-min_dist/2, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # perturb points
    max_movement = (init_dist - min_dist)/2
    noise = np.random.uniform(low=-max_movement,
                              high=max_movement,
                              size=(len(coords), 2))
    coords += noise
    return coords


def generate_points(n, shape, min_dist):
    coords = []
    for i in range(n):
        while True:
            x = np.random.uniform(low=min_dist/2, high=shape[0]-min_dist/2)
            y = np.random.uniform(low=min_dist/2, high=shape[1]-min_dist/2)
            cur_coord = np.array([x, y])

            reject = False
            for coord in coords:
                dist = np.linalg.norm(coord - cur_coord)
                if dist < min_dist:
                    reject = True
                    break

            if not reject:
                coords.append(cur_coord)
                break
    return np.array(coords)


def main():
    w = 100
    h = 100
    diameter = 1.5
    red_num = 50
    black_num = 50
    trials = 10

    coords = generate_points(n=red_num+black_num, shape=(w, h), min_dist=diameter)
    dye_red = np.random.randint(low=0, high=red_num+black_num, size=red_num)
    colors = np.zeros(len(coords))
    colors[dye_red] = 1  # reds are 1s; blacks are 0s

    # plot
    plt.figure(figsize=(5, 5))
    plt.scatter(coords[:, 0], coords[:, 1], s=3)
    plt.show()


if __name__ == '__main__':
    main()
