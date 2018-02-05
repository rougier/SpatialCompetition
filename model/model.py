import numpy as np
import itertools


class Model:

    def __init__(self, param):

        np.random.seed(param.seed)

        self.n_positions = param.n_positions
        self.n_prices = param.n_prices
        self.t_max = param.t_max
        self.r = param.r
        self.p_min = param.p_min
        self.p_max = param.p_max

        self.strategies = np.array(
            list(itertools.product(range(self.n_positions), range(self.n_prices))),
            dtype=int
        )

        self.prices = np.linspace(self.p_min, self.p_max, self.n_prices)

        # Useful n
        self.n_strategies = len(self.strategies)
        self.idx_strategies = np.arange(self.n_strategies)

        # Prepare useful arrays
        self.n_customers = self.compute_n_customers()

    def compute_n_customers(self):

        z = np.zeros((self.n_positions, self.n_positions, 3), dtype=int)
        # Last parameter is idx0: n customers seeing only A,
        #                   idx1: n customers seeing only B,
        #                   idx2: customers seeing A and B,

        field_of_view = np.zeros((self.n_positions, 2))  # 2: min, max
        field_of_view[:] = [self.field_of_view(x) for x in range(self.n_positions)]

        for i, j in itertools.combinations_with_replacement(range(self.n_positions), r=2):

            for x in range(self.n_positions):

                see_firm_0 = field_of_view[x, 0] <= i <= field_of_view[x, 1]
                see_firm_1 = field_of_view[x, 0] <= j <= field_of_view[x, 1]

                if see_firm_0 and see_firm_1:
                    z[i, j, 2] += 1

                elif see_firm_0:
                    z[i, j, 0] += 1

                elif see_firm_1:
                    z[i, j, 1] += 1

            z[j, i, 0] = z[i, j, 1]
            z[j, i, 1] = z[i, j, 0]
            z[j, i, 2] = z[i, j, 2]

        return z

    def field_of_view(self, x):

        r = int(self.r * self.n_positions)

        field_of_view = [
            max(x - r, 0),
            min(x + r, self.n_positions - 1)
        ]

        return field_of_view

    def profits_given_position_and_price(self, move0, move1):

        pos0, price0 = self.strategies[move0, :]
        pos1, price1 = self.strategies[move1, :]

        n_customers = np.zeros(2)
        n_customers[:] = self.n_customers[pos0, pos1, :2]

        to_share = self.n_customers[pos0, pos1, 2]

        if to_share > 0:

            if price0 == price1:
                n_customers[:] += to_share / 2

            else:
                n_customers[int(price1 < price0)] += to_share

        return n_customers * self.prices[np.array((price0, price1))]  # Price0, price1 are idx of prices

    def optimal_move(self, opp_move):

        exp_profits = np.zeros(self.n_strategies)

        for i in range(self.n_strategies):
            exp_profits[i] = self.profits_given_position_and_price(i, opp_move)[0]

        max_profits = max(exp_profits)

        idx = np.flatnonzero(exp_profits == max_profits)

        i = np.random.choice(idx)

        return i

    def run(self):

        # For recording
        positions = np.zeros((self.t_max, 2), dtype=int)
        prices = np.zeros((self.t_max, 2))
        profits = np.zeros((self.t_max, 2))

        moves = np.zeros(2, dtype=int)

        active = 0

        moves[:] = -99, np.random.randint(low=0, high=self.n_prices * self.n_positions)

        for t in range(self.t_max):

            passive = (active + 1) % 2

            moves[active] = self.optimal_move(moves[passive])

            positions[t, :] = self.strategies[moves, 0]
            prices[t, :] = self.prices[self.strategies[moves, 1]]
            profits[t, :] = self.profits_given_position_and_price(moves[0], moves[1])

            active = passive

        return positions, prices, profits
