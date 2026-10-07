# Modified from WorldEngine (https://github.com/OpenDriveLab/WorldEngine), licensed under Apache-2.0.
import json
import logging
import os
from typing import Dict, List, Tuple
from pathlib import Path
from omegaconf import DictConfig

from odyssey.envs.base_env import BaseEnv
from odyssey.runner.utils import RunnerReport

logger = logging.getLogger(__name__)


def save_runner_reports(reports: List[RunnerReport], output_dir: Path, report_file: str) -> Path:
    """
    Save runner reports to a JSON file.
    :param reports: List of RunnerReport from all simulations.
    :param output_dir: Directory to save the report file.
    :param report_file: Name of the report file (extension will be replaced with .json).
    :return: Path to the saved report file.
    """
    rows = []
    for report in reports:
        rows.append({
            'scenario_name': report.scenario_name,
            'log_name': report.log_name,
            'planner_name': report.planner_name,
            'succeeded': report.succeeded,
            'duration_s': round(report.end_time - report.start_time, 3) if report.end_time else None,
            'error_message': report.error_message,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / Path(report_file).with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved runner report ({len(reports)} entries) to {report_path}")
    return report_path

def run_simulation(sim_env: BaseEnv, exit_on_failure: bool = False) -> List[RunnerReport]:
    """
    Proxy for calling simulation.
    :param sim_env: An environment which will execute all batched simulations.
    :param exit_on_failure: If true, raises an exception when the simulation fails.
    :return report for the simulation.
    """
    try:
        reports = sim_env.run()
    finally:
        if os.environ.get('ODYSSEY_RUNTIME_PROFILE'):
            from odyssey_runtime.session import close_session
            close_session()  # Drain recording errors before any success flag.

    if os.environ.get('ODYSSEY_RUNTIME_PROFILE') and (not reports or not all(r.succeeded for r in reports)):
        return reports

    # Completion flag file
    flag_file_path = Path(sim_env.config.output_dir) / "simulation_completed.flag"
    flag_file_path.parent.mkdir(parents=True, exist_ok=True)
    flag_file_path.touch()
    logger.info(f"Generated completion flag file: {flag_file_path}")

    return reports

def run_runners(envs: List[BaseEnv], cfg: DictConfig) -> None:
    """
    Run a list of runners, one after another in this process.
    :param envs: A list of envs.
    :param cfg: Hydra config.
    """
    assert len(envs) > 0, 'No environments found to simulate!'

    logger.info('Executing runners...')
    reports = [run_simulation(env, cfg.exit_on_failure) for env in envs]
    # Flatten the list of lists
    reports = [report for sublist in reports for report in sublist]

    # Store the results in a dictionary so we can easily store error tracebacks in the next step, if needed
    results: Dict[Tuple[str, str, str], RunnerReport] = {
        (report.scenario_name, report.planner_name, report.log_name): report for report in reports
    }

    # Notify user about the result of simulations
    failed_simulations = str()
    number_of_successful = 0
    number_of_failures = 0
    runner_reports: List[RunnerReport] = list(results.values())
    for result in runner_reports:
        if result.succeeded:
            number_of_successful += 1
        else:
            if result.error_message is not None:
                number_of_failures += 1
                logger.warning("Failed Simulation.\n '%s'", result.error_message)
                failed_simulations += f"[{result.log_name}, {result.scenario_name}] \n"

    logger.info(f"Number of successful simulations: {number_of_successful}")
    logger.info(f"Number of failed simulations: {number_of_failures}")

    # Print out all failed simulation unique identifier
    if number_of_failures > 0:
        logger.info(f"Failed simulations [log, token]:\n{failed_simulations}")

    logger.info('Finished executing runners!')

    save_runner_reports(runner_reports, Path(cfg.output_dir), cfg.runner_report_file)
