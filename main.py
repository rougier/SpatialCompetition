import multiprocessing as mlt
import tqdm
import os

import model
import analysis
import backup
import parameters

import argparse


def run(param):

    m = model.Model(param)
    positions, prices, profits = m.run()
    bkp = backup.RunBackup(parameters=param, positions=positions, prices=prices, profits=profits)
    return bkp


def produce_data(parameters_file, data_file):

    json_parameters = parameters.load(parameters_file)

    pool_parameters = parameters.extract_parameters(json_parameters)

    pool = mlt.Pool()

    backups = []

    for bkp in tqdm.tqdm(
            pool.imap_unordered(run, pool_parameters),
            total=len(pool_parameters)):
        backups.append(bkp)

    pool_backup = backup.PoolBackup(parameters=json_parameters, backups=backups)
    pool_backup.save(data_file)

    return pool_backup


def get_files_names(condition):

    parameters_file = "parameters/json/{}.json".format(condition)
    data_file = {
        "pickle": "data/pickle/{}.p".format(condition),
        "json": "data/json/{}.json".format(condition)
    }

    if condition == 'pool':
        fig_names = {
            "main": "data/figs/{}.pdf".format(condition)
        }

    else:
        fig_names = {
            "eeg_like": "data/figs/eeg_like_{}.pdf".format(condition),
            "positions": "data/figs/positions_{}.pdf".format(condition)
        }

    return parameters_file, data_file, fig_names


def data_already_produced(data_file):
    return os.path.exists(data_file["json"]) and os.path.exists(data_file["pickle"])


def terminal_msg(condition, parameters_file, data_file, figure_files):

    print("\n************ For '{}' results **************".format(condition))
    print()
    print("Parameters file used is: '{}'\n".format(parameters_file))
    print("Data files are:\n"
          "* '{}' for parameters\n"
          "* '{}' for data itself\n".format(parameters_file, data_file["json"], data_file["pickle"]))
    print("Figures files are:")
    for file in figure_files.values():
        print("* '{}'".format(file))
    print()


def a_priori():

    figure_files = {
        "targetable_consumers": "data/figs/targetable_consumers.pdf",
        "captive_consumers_25": "data/figs/captive_consumers_25.pdf",
        "captive_consumers_50": "data/figs/captive_consumers_50.pdf",
        "captive_consumers_75": "data/figs/captive_consumers_75.pdf"
    }

    analysis.a_priori.targetable_consumers(figure_files["targetable_consumers"])
    analysis.a_priori.captive_consumers(0.25, figure_files["captive_consumers_25"])
    analysis.a_priori.captive_consumers(0.50, figure_files["captive_consumers_50"])
    analysis.a_priori.captive_consumers(0.75, figure_files["captive_consumers_75"])

    print("\n************ For 'a priori' analysis **************\n")
    print("Figures files are:")
    for file in figure_files.values():
        print("* '{}'".format(file))
    print()


def pooled_data(args):

    condition = "pool"

    parameters_file, data_file, fig_files = get_files_names(condition)

    if not data_already_produced(data_file) or args.force:
        pool_backup = produce_data(parameters_file, data_file)

    else:
        pool_backup = backup.PoolBackup.load(data_file["pickle"])

    analysis.pool.distance_over_fov(pool_backup=pool_backup, fig_name=fig_files["main"])

    terminal_msg(condition, parameters_file, data_file, fig_files)


def individual_data(args):

    for condition in ("75", "50", "25"):

        parameters_file, data_file, fig_files = get_files_names(condition)

        if not data_already_produced(data_file) or args.force:

            json_parameters = parameters.load(parameters_file)
            param = parameters.extract_parameters(json_parameters)
            run_backup = run(param)
            run_backup.save(data_file)

        else:
            run_backup = backup.RunBackup.load(data_file["pickle"])

        analysis.separate.eeg_like(backup=run_backup, fig_name=fig_files["eeg_like"])
        analysis.separate.pos_firmA_over_pos_firmB(backup=run_backup, fig_name=fig_files["positions"])

        terminal_msg(condition, parameters_file, data_file, fig_files)


def main(args):

    if args.new:
        args.force = True
        parameters.generate_new_parameters_files()

    if args.pooled or (not args.individual and not args.a_priori):
        pooled_data(args)

    if args.individual or (not args.pooled and not args.a_priori):
        individual_data(args)

    if args.a_priori or (not args.individual and not args.pooled):
        a_priori()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Produce figures.')
    parser.add_argument('-f', '--force', action="store_true", default=False,
                        help="Re-run simulations")
    parser.add_argument('-n', '--new', action="store_true", default=False,
                        help="Generate new parameters files")
    parser.add_argument('-i', '--individual', action="store_true", default=False,
                        help="Do figures ONLY for individual results")
    parser.add_argument('-p', '--pooled', action="store_true", default=False,
                        help="Do figures ONLY for pooled results")
    parser.add_argument('-a', '--a_priori', action="store_true", default=False,
                        help="Do figures ONLY for a priori analysis")
    parsed_args = parser.parse_args()

    main(parsed_args)



