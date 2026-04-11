"""CLI entry point for smoltrain."""
import click
from . import config as cfg_mod


@click.group()
def main():
    """smoltrain — tiny text classifier trainer."""
    pass


@main.command()
@click.argument("name")
@click.option("--classes", required=True, help="Comma-separated class labels")
@click.option("--goal", required=True, help="Description of the classification task")
def new(name, classes, goal):
    """Create a new classification task."""
    cfg_mod.new_task(name, classes.split(","), goal)
    click.echo(f"created task '{name}' at {cfg_mod.config_path(name)}")


@main.command()
@click.argument("name")
@click.option("--count", default=None, type=int, help="Examples per class (overrides config)")
def gen(name, count):
    """Generate training examples via supervisor LLM."""
    from .gen import run
    cfg = cfg_mod.load(name)
    if count:
        cfg.n_per_class = count
    run(cfg)


@main.command()
@click.argument("name")
def train(name):
    """Fine-tune the model on the dataset."""
    from .train import run
    run(cfg_mod.load(name))


@main.command()
@click.argument("name")
def export(name):
    """Export fine-tuned model to ONNX INT8."""
    from .export import run
    run(cfg_mod.load(name))


@main.command()
@click.argument("name")
def serve(name):
    """Run classifier daemon on Unix socket."""
    from .serve import run
    run(cfg_mod.load(name))


@main.command()
@click.argument("name")
@click.argument("input")
def classify(name, input):
    """Classify a single input text."""
    from .classify import OnnxClassifier
    c = OnnxClassifier(cfg_mod.load(name))
    click.echo(c.classify(input))


@main.command("pipeline", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.argument("pipeline_args", nargs=-1, type=click.UNPROCESSED)
def pipeline_cmd(pipeline_args):
    """Run full CharCNN pipeline: merge -> train -> eval -> export -> latency."""
    import sys
    from smoltrain import pipeline
    sys.argv = [sys.argv[0]] + list(pipeline_args)
    pipeline.main()


@main.command("run")
@click.argument("name")
@click.option("--count", default=None, type=int, help="Examples per class (overrides config)")
@click.option("--skip-gen", is_flag=True, help="Skip gen if dataset exists")
def run_pipeline(name, count, skip_gen):
    """Full pipeline: gen → train → export."""
    from . import gen as gen_mod
    from . import train as train_mod
    from . import export as export_mod

    cfg = cfg_mod.load(name)
    if count:
        cfg.n_per_class = count

    ds = cfg_mod.dataset_path(name)
    if skip_gen and ds.exists():
        click.echo("skipping gen (dataset exists)")
    else:
        gen_mod.run(cfg)

    train_mod.run(cfg)
    export_mod.run(cfg)
    click.echo("done. run: smoltrain serve " + name)
