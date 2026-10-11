"""Refresh configured human inputs and geometry previews, without simulation."""
import hashlib
from pathlib import Path
from annotate_transfer_demos import prepare, read_json, write_json
from plan_transfer_rollouts import plan
from retarget_libero_object import TASK, load_transport

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/data_003_rollouts.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    config = read_json(path)
    if config.get("schema_version") != 1 or config.get("task") != TASK:
        raise ValueError("Unsupported manifest schema/task")
    if config.get("benchmark") != "libero_spatial" or config.get("task_index") != 0:
        raise ValueError("This runner supports only the validated bowl-to-plate task")
    if config.get("rollouts_per_demo") != 1 or config.get("automatic_retries") is not False:
        raise ValueError("Exactly one rollout per demo, with no automatic retries, is required")
    if config.get("baseline_episode_ids") != ["episode_001", "episode_002"]:
        raise ValueError("Select exactly the two requested successful baseline episodes")
    demos = config.get("demos", [])
    if not demos:
        raise ValueError("No demonstrations configured")
    for key in ("demo_id", "episode_id", "prepared_demo"):
        if len({d[key] for d in demos}) != len(demos):
            raise ValueError("Duplicate " + key)
    for demo in demos:
        if demo["direction"] not in ("object -> target", "target -> object"):
            raise ValueError("Unresolved direction: " + demo["demo_id"])
        if demo["episode_id"] in config["baseline_episode_ids"]:
            raise ValueError("New episode collides with baseline")
        budget=demo.get("transport_step_budget")
        if not isinstance(budget,int) or isinstance(budget,bool) or budget<=0:
            raise ValueError("Per-demo transport_step_budget must be a positive integer")
        for key in ("source_demo", "annotation_file", "prepared_demo"):
            value = Path(demo[key])
            if value.is_absolute() or ".." in value.parts:
                raise ValueError("Manifest paths must be portable repository-relative paths")
    return config


def claim_attempt(directory):
    """Exclusive directory creation is the durable one-attempt guard."""
    try:
        Path(directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False
    return True


def prepare_bundle(config_path=DEFAULT_CONFIG):
    config_path = Path(config_path)
    config = load_config(config_path)
    reports = []
    for entry in config["demos"]:
        output = Path(entry["prepared_demo"])
        reasons = []
        validation = None
        preview = None
        try:
            validation = prepare(Path(entry["source_demo"]), Path(entry["annotation_file"]), output)
            if validation["demo_id"] != entry["demo_id"] or validation["direction"] != entry["direction"]:
                raise ValueError("Manifest/source identity or direction mismatch")
            reasons.extend(validation["blocking_reasons"])
            annotation = read_json(entry["annotation_file"])
            if config.get("require_manual_transport_bounds", False):
                for key in ("transport_start_time_seconds", "transport_end_time_seconds"):
                    if annotation.get(key) is None:
                        reasons.append("Manual review required: set " + key + " in " + entry["annotation_file"])
            if validation["suitable_for_retargeting"]:
                preview = plan(output, config["robot_config"], config["mapping_config"], config["nominal_preflight"], entry["transport_step_budget"])
                reasons.extend(preview["reasons"])
                human = load_transport(output/"processed_demo.csv", output/"metadata.json",
                                       read_json(config["mapping_config"])["max_source_gap_s"])
                sample_count = len(human[0])
            else:
                sample_count = 0
            provenance = dict(source_video=read_json(output/"metadata.json")["source_video"],
                source_demo=entry["source_demo"], annotation_file=entry["annotation_file"],
                source_trajectory_sha256=digest(Path(entry["source_demo"])/"processed_demo.csv"),
                annotation_sha256=digest(entry["annotation_file"]),
                prepared_trajectory_sha256=digest(output/"processed_demo.csv"),
                prepared_metadata_sha256=digest(output/"metadata.json"),
                manifest_sha256=digest(config_path), robot_config_sha256=digest(config["robot_config"]),
                mapping_config_sha256=digest(config["mapping_config"]))
            robot_settings=read_json(config["robot_config"])
            robot_settings["phase_max_steps"]["TRANSPORT"]=entry["transport_step_budget"]
            record = dict(**entry, robot_settings=robot_settings, mapping_settings=read_json(config["mapping_config"]), schema_version=1, task=config["task"], benchmark=config["benchmark"],
                task_index=config["task_index"], seed=config["seed"], requested_rollouts=1,
                automatic_retries=False, ready_for_rollout=not reasons, blocking_reasons=reasons,
                validation=validation, mapping_preview=preview, transport_samples=sample_count,
                provenance=provenance,
                mapping_semantics="Align the source demo start-to-end axis to bowl-start-to-plate; preserve sample order, lateral path and lift residual. Robot contact phases remain scripted.")
        except (ValueError, FileNotFoundError, KeyError) as exc:
            record = dict(**entry, ready_for_rollout=False,
                          blocking_reasons=[type(exc).__name__ + ": " + str(exc)],
                          validation=validation, mapping_preview=preview)
        write_json(output/"retargeting_input.json", record)
        reports.append(record)
    write_json(Path(config["prepared_root"])/"validation.json", [r["validation"] for r in reports if r["validation"]])
    write_json(Path(config["prepared_root"])/"mapping_previews.json", [r["mapping_preview"] for r in reports if r["mapping_preview"]])
    result = dict(schema_version=1, config=config_path.as_posix(), task=config["task"],
                  prepared_demos=len(reports), ready_demos=sum(r["ready_for_rollout"] for r in reports),
                  requested_rollouts_per_demo=1, automatic_retries=False, actual_rollouts=0,
                  demos=reports)
    write_json(Path(config["prepared_root"])/"rollout_inputs.json", result)
    return config, result
