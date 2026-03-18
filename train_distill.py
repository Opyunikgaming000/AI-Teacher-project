import argparse
import copy
import gc
import glob
import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Optional
from urllib import error, request

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, models, transforms

from student_model import build_student_model_200mb


PROFESSION_TOPICS = [
    "Accountant", "Actor", "Actuary", "Acupuncturist", "Architect", "Artist", "Animator", "Archaeologist",
    "Astronomer", "Author", "Aeromedical Technician", "AI Engineer", "Baker", "Banker", "Barber", "Barista",
    "Biologist", "Blacksmith", "Bodyguard", "Bookkeeper", "Broker", "Builder", "Butcher", "Butler",
    "Biomedical Scientist", "Carpenter", "Cartographer", "Cashier", "Chef", "Chemist", "Civil Servant",
    "Coach", "Computer Programmer", "Copywriter", "Counselor", "Curator", "Cybersecurity Analyst", "Dancer",
    "Data Scientist", "Dentist", "Dermatologist", "Designer", "Detective", "Dietitian", "Diplomat", "Director",
    "Diver", "Doctor", "Driver", "Delivery Associate", "Ecologist", "Economist", "Editor", "Electrician",
    "Engineer", "Entrepreneur", "Entomologist", "Environmentalist", "Event Planner", "Executive", "Epidemiologist",
    "Farmer", "Fashion Designer", "Film Director", "Financial Advisor", "Firefighter", "Fisherman", "Florist",
    "Forensic Scientist", "Forester", "Flight Attendant", "Game Developer", "Gardener", "Geneticist", "Geographer",
    "Geologist", "Glazier", "Goldsmith", "Graphic Designer", "Guide", "Gynecologist", "Hairdresser", "Historian",
    "Horticulturist", "Hotel Manager", "Human Resources Manager", "Hydrologist", "Hypnotherapist", "HVAC Technician",
    "Illustrator", "Interior Designer", "Investment Banker", "Interpreter", "IT Specialist", "Investigator",
    "Immunologist", "Industrial Designer", "Janitor", "Jeweler", "Jockey", "Journalist", "Judge",
    "Justice of the Peace", "Joiner", "Keeper (Zoo/Museum)", "Kindergarten Teacher", "Kinesiologist", "Knitter",
    "Lawyer", "Lecturer", "Librarian", "Lifeguard", "Linguist", "Locksmith", "Logistician", "Lab Technician",
    "Landscape Architect", "Machinist", "Magician", "Maid", "Manager", "Marine Biologist", "Marketing Specialist",
    "Mason", "Mathematician", "Mechanic", "Meteorologist", "Model", "Musician", "Nanny", "Navigator",
    "Network Engineer", "Neurologist", "Newsreader", "Novelist", "Nurse", "Nutritionist", "Nuclear Physicist",
    "Oceanographer", "Optician", "Orthodontist", "Osteopath", "Office Assistant", "Occupational Therapist",
    "Oncologist", "Painter", "Paleontologist", "Paramedic", "Pathologist", "Pharmacist", "Photographer",
    "Physician", "Physicist", "Pilot", "Plumber", "Police Officer", "Politician", "Psychologist",
    "Quality Control Inspector", "Quantitative Analyst", "Quarry Worker", "Quantum Physicist", "Radio Jockey (RJ)",
    "Radiologist", "Real Estate Agent", "Receptionist", "Researcher", "Reporter", "Robotics Engineer", "Roofer",
    "Sailor", "Salesperson", "Scientist", "Sculptor", "Secretary", "Security Guard", "Singer", "Social Worker",
    "Software Developer", "Statistician", "Surgeon", "Surveyor", "Tailor", "Teacher", "Technician", "Telemarketer",
    "Therapist", "Tour Guide", "Translator", "Truck Driver", "Tutor", "Typist", "Tax Consultant", "Umpire",
    "Underwriter", "UI/UX Designer", "Urologist", "Urban Planner", "Undertaker", "Valuer", "Veterinarian",
    "Video Editor", "Violinist", "Voice Actor", "Volunteer", "VFX Artist", "Waiter/Waitress", "Watchmaker",
    "Weaver", "Web Developer", "Welder", "Wildlife Biologist", "Writer", "Woodworker", "Warehouse Manager",
    "X-ray Technician", "Xenobiologist", "Xylophonist", "Yardmaster", "Yoga Instructor", "YouTuber", "Youth Worker",
    "Zoologist", "Zookeeper", "Zitherist",
]


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_acc: float = 0.0


@dataclass
class StrictCheckResult:
    perfect_match: bool
    batch_accuracy: float
    student_preds: list[int]
    teacher_preds: list[int]


class DistillationLoss(nn.Module):
    """
    Combines hard-label CE loss + soft-target KD loss.
    """

    def __init__(self, alpha: float = 0.5, temperature: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hard_loss = self.ce(student_logits, labels)

        t = self.temperature
        student_log_probs = torch.log_softmax(student_logits / t, dim=1)
        teacher_probs = torch.softmax(teacher_logits / t, dim=1)
        soft_loss = self.kl(student_log_probs, teacher_probs) * (t * t)

        return self.alpha * soft_loss + (1.0 - self.alpha) * hard_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Knowledge distillation with auto checkpoint resume")

    parser.add_argument("--train-dir", type=str, required=True, help="Path to ImageFolder train directory")
    parser.add_argument("--val-dir", type=str, default="", help="Path to ImageFolder val directory")
    parser.add_argument("--num-classes", type=int, required=True, help="Number of target classes")

    parser.add_argument("--teacher-arch", type=str, default="resnet50", help="torchvision teacher architecture")
    parser.add_argument("--teacher-checkpoint", type=str, default="", help="Teacher checkpoint (.pth)")
    parser.add_argument("--teacher-pretrained", action="store_true", help="Use torchvision pretrained teacher")

    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.7, help="KD soft loss weight")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--strict-teacher", action="store_true", help="Enable strict 100% topic gating")
    parser.add_argument("--penalty-base", type=float, default=1.0, help="Initial strict retry loss multiplier")
    parser.add_argument("--penalty-growth", type=float, default=1.5, help="Multiplier for every failed strict retry")
    parser.add_argument(
        "--max-retries-per-batch",
        type=int,
        default=0,
        help="0 means unlimited; otherwise retries allowed for each strict batch",
    )
    parser.add_argument(
        "--topic-curriculum",
        type=str,
        default="professions",
        choices=("professions", "dataset-order"),
        help="Topic order in strict mode",
    )
    parser.add_argument(
        "--career-submodules",
        action="store_true",
        help="Train strict topics in letter-wise sub-modules to avoid mixing",
    )
    parser.add_argument(
        "--submodule-order",
        type=str,
        default="A,B",
        help="Comma-separated letter order for career sub-modules (e.g., A,B,C)",
    )
    parser.add_argument("--homework-log", type=str, default="homework_master.jsonl")
    parser.add_argument("--mistake-log", type=str, default="homework_mistakes.jsonl")
    parser.add_argument("--llama-endpoint", type=str, default="", help="Optional Llama teacher HTTP endpoint")
    parser.add_argument("--llama-model", type=str, default="llama-3.1-405b")
    parser.add_argument("--llama-api-key", type=str, default="", help="Optional bearer token for llama endpoint")

    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Optional explicit checkpoint path to resume")
    parser.add_argument("--disable-auto-resume", action="store_true", help="Disable resume from latest checkpoint")
    parser.add_argument("--ckpt-interval-min", type=float, default=10.0, help="Checkpoint interval in minutes")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_temp_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def append_jsonl(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def normalize_topic_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def topic_first_letter(name: str) -> str:
    stripped = name.strip()
    return stripped[0].upper() if stripped else "#"


def build_topic_order(class_names: list[str], mode: str) -> list[str]:
    if mode == "dataset-order":
        return class_names

    indexed = {normalize_topic_name(x): x for x in class_names}
    ordered: list[str] = []
    for topic in PROFESSION_TOPICS:
        key = normalize_topic_name(topic)
        if key in indexed:
            ordered.append(indexed[key])

    for cls in class_names:
        if cls not in ordered:
            ordered.append(cls)
    return ordered


def maybe_query_llama_logic(
    endpoint: str,
    model_name: str,
    api_key: str,
    topic: str,
    teacher_pred: int,
) -> str:
    if not endpoint:
        return ""

    prompt = (
        f"Career task topic: {topic}. "
        f"Expected class index from strict teacher logic: {teacher_pred}. "
        "Return one line with exact reasoning rule and final class index."
    )

    body = json.dumps({"model": model_name, "prompt": prompt}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=15) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
            return text[:1000]
    except (error.URLError, TimeoutError, ValueError):
        return ""


def create_torchvision_model(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    if hasattr(models, "get_model"):
        kwargs = {"num_classes": num_classes}
        if pretrained:
            kwargs["weights"] = "DEFAULT"
        else:
            kwargs["weights"] = None
        return models.get_model(name, **kwargs)

    if not hasattr(models, name):
        raise ValueError(f"Unknown torchvision model: {name}")

    builder = getattr(models, name)
    sig = inspect.signature(builder)
    kwargs = {}

    if "weights" in sig.parameters:
        kwargs["weights"] = "DEFAULT" if pretrained else None
    elif "pretrained" in sig.parameters:
        kwargs["pretrained"] = pretrained

    if "num_classes" in sig.parameters:
        kwargs["num_classes"] = num_classes

    return builder(**kwargs)


def unwrap_state_dict(state: dict) -> dict:
    for key in ("state_dict", "model_state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            return state[key]
    return state


def load_teacher(
    arch: str,
    checkpoint_path: str,
    pretrained: bool,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    teacher = create_torchvision_model(arch, pretrained=pretrained, num_classes=num_classes)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location="cpu")
        teacher.load_state_dict(unwrap_state_dict(state), strict=False)

    teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def make_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, Optional[DataLoader], datasets.ImageFolder]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tfms)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = None
    if args.val_dir and os.path.isdir(args.val_dir):
        val_ds = datasets.ImageFolder(args.val_dir, transform=val_tfms)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, train_ds


def make_topic_loaders(
    train_ds: datasets.ImageFolder,
    args: argparse.Namespace,
    class_names: list[str],
) -> list[tuple[str, DataLoader]]:
    topic_order = build_topic_order(class_names, args.topic_curriculum)
    target_to_indices: dict[int, list[int]] = {idx: [] for idx in range(len(class_names))}

    for idx, (_, label) in enumerate(train_ds.samples):
        target_to_indices[label].append(idx)

    topic_loaders: list[tuple[str, DataLoader]] = []
    for topic_name in topic_order:
        label = train_ds.class_to_idx[topic_name]
        indices = target_to_indices.get(label, [])
        if not indices:
            continue

        subset = Subset(train_ds, indices)
        topic_loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        topic_loaders.append((topic_name, topic_loader))

    return topic_loaders


def parse_submodule_order(order_csv: str) -> list[str]:
    letters: list[str] = []
    for raw in order_csv.split(","):
        token = raw.strip().upper()
        if not token:
            continue
        if len(token) != 1 or not token.isalpha():
            raise ValueError(f"Invalid submodule letter '{raw}'. Use format like A,B,C")
        if token not in letters:
            letters.append(token)
    return letters


def make_submodule_topic_loaders(
    train_ds: datasets.ImageFolder,
    args: argparse.Namespace,
    class_names: list[str],
) -> list[tuple[str, list[tuple[str, DataLoader]]]]:
    topic_loaders = make_topic_loaders(train_ds, args, class_names)
    by_letter: dict[str, list[tuple[str, DataLoader]]] = {}

    for topic_name, loader in topic_loaders:
        letter = topic_first_letter(topic_name)
        by_letter.setdefault(letter, []).append((topic_name, loader))

    ordered_letters = parse_submodule_order(args.submodule_order)
    submodules: list[tuple[str, list[tuple[str, DataLoader]]]] = []
    for letter in ordered_letters:
        topics = by_letter.get(letter, [])
        if topics:
            submodules.append((letter, topics))

    if not submodules:
        raise RuntimeError(
            "No topics found for selected submodule letters. "
            "Check class names and --submodule-order."
        )
    return submodules


def latest_checkpoint_path(checkpoint_dir: str) -> str:
    paths = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not paths:
        return ""
    paths.sort(key=os.path.getmtime)
    return paths[-1]


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    train_state: TrainState,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "epoch": train_state.epoch,
        "global_step": train_state.global_step,
        "best_val_acc": train_state.best_val_acc,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> TrainState:
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    if "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    return TrainState(
        epoch=int(ckpt.get("epoch", 0)),
        global_step=int(ckpt.get("global_step", 0)),
        best_val_acc=float(ckpt.get("best_val_acc", 0.0)),
    )


def evaluate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    model.train()
    return (correct / total) if total else 0.0


def strict_match_check(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> StrictCheckResult:
    student_preds = student_logits.argmax(dim=1)
    teacher_preds = teacher_logits.argmax(dim=1)

    teacher_and_label_match = (student_preds == teacher_preds) & (student_preds == labels)
    batch_acc = float(teacher_and_label_match.float().mean().item())

    return StrictCheckResult(
        perfect_match=(batch_acc == 1.0),
        batch_accuracy=batch_acc,
        student_preds=student_preds.detach().cpu().tolist(),
        teacher_preds=teacher_preds.detach().cpu().tolist(),
    )


def clone_training_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
) -> tuple[dict, dict, dict, dict]:
    return (
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
        copy.deepcopy(scaler.state_dict()),
    )


def restore_training_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    snapshot: tuple[dict, dict, dict, dict],
) -> None:
    model_state, optim_state, sched_state, scaler_state = snapshot
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optim_state)
    scheduler.load_state_dict(sched_state)
    scaler.load_state_dict(scaler_state)


def maybe_save_timed_checkpoint(
    args: argparse.Namespace,
    epoch: int,
    batch_idx: int,
    student: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    train_state: TrainState,
    last_timed_ckpt_at: float,
    ckpt_interval_seconds: float,
) -> float:
    now = time.time()
    if now - last_timed_ckpt_at < ckpt_interval_seconds:
        return last_timed_ckpt_at

    timed_ckpt = os.path.join(
        args.checkpoint_dir,
        f"timed_e{epoch:04d}_b{batch_idx:05d}_step{train_state.global_step:09d}.pth",
    )
    train_state.epoch = epoch
    save_checkpoint(timed_ckpt, student, optimizer, scheduler, scaler, train_state)
    save_checkpoint(
        os.path.join(args.checkpoint_dir, "latest.pth"),
        student,
        optimizer,
        scheduler,
        scaler,
        train_state,
    )
    print(f"[Checkpoint] Timed save: {timed_ckpt}")
    return now


def train_strict_topic(
    topic_name: str,
    topic_loader: DataLoader,
    epoch: int,
    args: argparse.Namespace,
    student: nn.Module,
    teacher: nn.Module,
    criterion: DistillationLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    train_state: TrainState,
    device: torch.device,
    ckpt_interval_seconds: float,
    last_timed_ckpt_at: float,
) -> tuple[float, int, float]:
    running_loss = 0.0
    committed_batches = 0

    for batch_idx, (images, labels) in enumerate(topic_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        retry = 0
        penalty = max(args.penalty_base, 1.0)

        while True:
            snapshot = clone_training_state(student, optimizer, scheduler, scaler)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                student_logits = student(images)
                with torch.no_grad():
                    teacher_logits = teacher(images)
                base_loss = criterion(student_logits, teacher_logits, labels)
                loss = base_loss * penalty

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                post_student_logits = student(images)
                check = strict_match_check(post_student_logits, teacher_logits, labels)

            if check.perfect_match:
                train_state.global_step += 1
                running_loss += float(loss.item())
                committed_batches += 1

                llama_trace = maybe_query_llama_logic(
                    args.llama_endpoint,
                    args.llama_model,
                    args.llama_api_key,
                    topic_name,
                    check.teacher_preds[0],
                )

                append_jsonl(
                    args.homework_log,
                    {
                        "time": int(time.time()),
                        "topic": topic_name,
                        "epoch": epoch,
                        "batch": batch_idx,
                        "retry_count": retry,
                        "penalty_used": penalty,
                        "accuracy": 1.0,
                        "student_preds": check.student_preds,
                        "teacher_preds": check.teacher_preds,
                        "llama_trace": llama_trace,
                    },
                )

                last_timed_ckpt_at = maybe_save_timed_checkpoint(
                    args,
                    epoch,
                    batch_idx,
                    student,
                    optimizer,
                    scheduler,
                    scaler,
                    train_state,
                    last_timed_ckpt_at,
                    ckpt_interval_seconds,
                )
                break

            append_jsonl(
                args.mistake_log,
                {
                    "time": int(time.time()),
                    "topic": topic_name,
                    "epoch": epoch,
                    "batch": batch_idx,
                    "retry": retry,
                    "penalty": penalty,
                    "accuracy": check.batch_accuracy,
                    "student_preds": check.student_preds,
                    "teacher_preds": check.teacher_preds,
                },
            )

            restore_training_state(student, optimizer, scheduler, scaler, snapshot)
            optimizer.zero_grad(set_to_none=True)

            del snapshot
            del student_logits
            del teacher_logits
            del post_student_logits
            clean_temp_memory(device)

            retry += 1
            penalty = penalty * max(args.penalty_growth, 1.01)

            if args.max_retries_per_batch > 0 and retry > args.max_retries_per_batch:
                raise RuntimeError(
                    f"Strict topic '{topic_name}' did not reach 100% match in batch {batch_idx}. "
                    f"Retries exhausted at {args.max_retries_per_batch}."
                )

    return running_loss, committed_batches, last_timed_ckpt_at


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, train_ds = make_dataloaders(args)
    class_names = list(train_ds.class_to_idx.keys())

    student = build_student_model_200mb(num_classes=args.num_classes).to(device)
    teacher = load_teacher(
        arch=args.teacher_arch,
        checkpoint_path=args.teacher_checkpoint,
        pretrained=args.teacher_pretrained,
        num_classes=args.num_classes,
        device=device,
    )

    criterion = DistillationLoss(alpha=args.alpha, temperature=args.temperature)
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    train_state = TrainState(epoch=0, global_step=0, best_val_acc=0.0)

    resume_path = args.resume.strip()
    if not resume_path and not args.disable_auto_resume:
        resume_path = latest_checkpoint_path(args.checkpoint_dir)

    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        train_state = load_checkpoint(
            resume_path,
            student,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        # Continue from the next epoch after the checkpointed epoch.
        train_state.epoch += 1

    ckpt_interval_seconds = max(args.ckpt_interval_min, 0.1) * 60.0
    last_timed_ckpt_at = time.time()

    try:
        for epoch in range(train_state.epoch, args.epochs):
            student.train()
            running_loss = 0.0
            batch_count = 0

            if args.strict_teacher:
                if args.career_submodules:
                    submodules = make_submodule_topic_loaders(train_ds, args, class_names)
                    for module_letter, module_topics in submodules:
                        print(f"[Sub-Module] Starting letter {module_letter}")
                        for topic_name, topic_loader in module_topics:
                            topic_loss, topic_batches, last_timed_ckpt_at = train_strict_topic(
                                topic_name=topic_name,
                                topic_loader=topic_loader,
                                epoch=epoch,
                                args=args,
                                student=student,
                                teacher=teacher,
                                criterion=criterion,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler,
                                train_state=train_state,
                                device=device,
                                ckpt_interval_seconds=ckpt_interval_seconds,
                                last_timed_ckpt_at=last_timed_ckpt_at,
                            )
                            running_loss += topic_loss
                            batch_count += topic_batches
                            print(f"[Strict Topic] {topic_name} mastered with 100% batch matches")
                        print(f"[Sub-Module] Completed letter {module_letter}")
                else:
                    topic_loaders = make_topic_loaders(train_ds, args, class_names)
                    for topic_name, topic_loader in topic_loaders:
                        topic_loss, topic_batches, last_timed_ckpt_at = train_strict_topic(
                            topic_name=topic_name,
                            topic_loader=topic_loader,
                            epoch=epoch,
                            args=args,
                            student=student,
                            teacher=teacher,
                            criterion=criterion,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            train_state=train_state,
                            device=device,
                            ckpt_interval_seconds=ckpt_interval_seconds,
                            last_timed_ckpt_at=last_timed_ckpt_at,
                        )
                        running_loss += topic_loss
                        batch_count += topic_batches
                        print(f"[Strict Topic] {topic_name} mastered with 100% batch matches")
            else:
                for batch_idx, (images, labels) in enumerate(train_loader):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                        student_logits = student(images)
                        with torch.no_grad():
                            teacher_logits = teacher(images)
                        loss = criterion(student_logits, teacher_logits, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()
                    batch_count += 1
                    train_state.global_step += 1

                    last_timed_ckpt_at = maybe_save_timed_checkpoint(
                        args,
                        epoch,
                        batch_idx,
                        student,
                        optimizer,
                        scheduler,
                        scaler,
                        train_state,
                        last_timed_ckpt_at,
                        ckpt_interval_seconds,
                    )

            scheduler.step()

            avg_loss = running_loss / max(batch_count, 1)
            print(f"Epoch [{epoch + 1}/{args.epochs}] loss={avg_loss:.4f}")

            val_acc = 0.0
            if val_loader is not None:
                val_acc = evaluate(student, val_loader, device)
                print(f"Validation accuracy: {val_acc * 100:.2f}%")

            train_state.epoch = epoch
            if val_acc > train_state.best_val_acc:
                train_state.best_val_acc = val_acc
                save_checkpoint(
                    os.path.join(args.checkpoint_dir, "best.pth"),
                    student,
                    optimizer,
                    scheduler,
                    scaler,
                    train_state,
                )

            epoch_ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1:04d}.pth")
            save_checkpoint(epoch_ckpt, student, optimizer, scheduler, scaler, train_state)
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "latest.pth"),
                student,
                optimizer,
                scheduler,
                scaler,
                train_state,
            )

    except KeyboardInterrupt:
        interrupt_ckpt = os.path.join(args.checkpoint_dir, "interrupted_latest.pth")
        save_checkpoint(interrupt_ckpt, student, optimizer, scheduler, scaler, train_state)
        print(f"Training interrupted. Saved checkpoint: {interrupt_ckpt}")
        raise


if __name__ == "__main__":
    main()
