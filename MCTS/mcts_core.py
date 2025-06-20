import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, PreTrainedTokenizer
import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm, trange
import re

from prm_config import OmegaPRMConfig

@dataclass
class Node:
    """A node stores the partial solution string and MCTS statistics."""
    state: str                     # concatenated steps so far (can be empty)
    parent: Optional["Node"]
    prior: float                   # optional prior from policy – not used here
    children: Dict[str, "Node"]    # action (next step) → Node

    n_visits: int = 0
    q_value: float = 0.0           # mean rollout success ratio
    correct_rollouts: int = 0       # cumulative successes
    total_rollouts: int = 0         # cumulative rollouts (denominator of MC)

    def ucb(self, cpuct: float, alpha: float, total_parent_visits: int) -> float:
        if self.n_visits == 0:
            return float("inf")  # force unseen nodes to be explored once
        exploration = cpuct * math.sqrt(math.log(total_parent_visits + 1) / (self.n_visits))
        return self.q_value + alpha * exploration


class MCTS:
    """MCTS driver for LLM‑based step‑wise mathematical reasoning."""
    STEP_PATTERN = re.compile(r"Step\s+\d+:")
    ANSWER_PATTERN = re.compile(r"Answer\s*:\s*(.+?)\s*(?:$|\n)")

    def __init__(self, config: "OmegaPRMConfig", golden_answers: Dict[str, str]):
        self.config = config
        self.golden_answers = golden_answers
        # Device & model ----------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # Generation configs ------------------------------------------------
        # Expansion: one step → we only need ~64 tokens max, sample top-k 8
        self.gen_cfg_expand = GenerationConfig(
            max_new_tokens=128,
            # top_k=5,
            do_sample=True,
            temperature=0.8,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Rollout: can be longer; keep top-k large to encourage diversity
        self.gen_cfg_rollout = GenerationConfig( 
            max_new_tokens=config.max_rollout_tokens, 
            # top_k=10,
            do_sample=True,
            temperature=0.8,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # Root placeholder (empty state).
        self.root = Node(state="", parent=None, prior=0.0, children={})
    
    # ---------------------------------------------------------------------
    # 1‑A. Low‑level helpers
    # ---------------------------------------------------------------------
    def _prompt_expand(self, question: str, partial_solution: str) -> str:
        if partial_solution:
            next_idx = self._next_step_idx(partial_solution)
            system = "You are a math‑problem expert. Generate **exactly one** next step following the numbered format \"Step k: ...\". Do NOT write more than one step or output the final answer directly. Never skip the step number formatand you MUST follow the given format."
            return (
                f"{system}\nProblem: {question}\n{partial_solution}\n"
                f"Step {next_idx}:"
            )
        # root
        return (
            "You are a math‑problem expert. Generate **exactly one** first step in the format \"Step 1: ...\". You must follow the format. Do NOT write more than one step or output the final answer directly. Never skip the step number format and you MUST follow the given format\n"
            f"Problem: {question}\nStep 1:"
        )

    def _prompt_rollout(self, question: str, partial_solution: str) -> str:
        intro = "You are a math‑problem expert. Continue the reasoning from the current step‑by‑step solution. You may write multiple additional steps \"Step k+1: ..., Step k+2:... \" with this format as needed to solve the problem. When the solution is complete, write a **single final line** beginning with \"Answer: \" followed by only the final answer. Do NOT add explanations, extra steps, or any trailing text after you reach the \"Answer: \". Strictly follow the given generation format during step-by-step reasoning."
        if partial_solution:
            next_idx = self._next_step_idx(partial_solution)
            return (
                f"{intro}\nProblem: {question}\n{partial_solution}\nStep {next_idx}:"
            )
        else:
            return (
                f"{intro}\nProblem: {question}\nStep 1:"
            )
    
    @staticmethod
    def _next_step_idx(solution: str) -> int:
        """Return index of the next step number."""
        matches = list(MCTS.STEP_PATTERN.finditer(solution))
        return len(matches) + 1

    def _extract_answer(self, text: str) -> Optional[str]:
        m = self.ANSWER_PATTERN.search(text)
        return m.group(1).strip() if m else None
    
    # ---------------------------------------------------------------------
    # 1‑B. Tree policy – selection & expansion
    # ---------------------------------------------------------------------
    def _select(self, node: Node) -> Node:
        """Traverse the tree until we hit a leaf (node without children)."""
        while node.children:
            # Choose child with maximal UCB score
            total = max(1, node.n_visits)
            best_action, node = max(
                node.children.items(),
                key=lambda kv: kv[1].ucb(self.config.cpuct, self.config.alpha, total),
            )
        return node
    
    def _split_steps(self, text: str) -> List[str]:
        """Turn 'Step i:' stream into a list of distinct step strings."""
        parts = self.STEP_PATTERN.split(text)
        headers = self.STEP_PATTERN.findall(text)
        steps = [h + p.strip() for h, p in zip(headers, parts[1:])]
        # Ensure each step ends with a newline for readability
        return [s if s.endswith("\n") else s + "\n" for s in steps if s]

    def _expand(self, node: Node, question: str):
        """Generate *top_k* candidate next steps from the language model."""
        if node is None:
            return
        prompt = self._prompt_expand(question, node.state)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, **self.gen_cfg_expand.to_dict())
        new_text = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # Split into candidate steps (may generate multiple Step k: blocks)
        steps = self._split_steps(new_text)
        if not steps:
            next_idx = self._next_step_idx(node.state)
            steps = [f"Step {next_idx}: {new_text.strip()}"]

        first_new_child = None
        for step in steps[: self.config.search_limit]:
            if step not in node.children:                 # 새로 본 step
                child_state = f"{node.state}{step}\n"
                child = Node(state=child_state,
                            parent=node,
                            prior=0.0,
                            children={})
                node.children[step] = child
                if first_new_child is None:               # 첫 신규 child 기억
                    first_new_child = child
        
        # print(f"[Expand] Node(partial_solution=\"{node.state}\") -> Generated steps: {steps}")
        # print(f"Children count: {len(node.children)}")
        return first_new_child

    # ---------------------------------------------------------------------
    # 1‑C. Simulation (rollout)
    # ---------------------------------------------------------------------
    def _rollout_from(self, node: Node, question: str) -> float:
        """Perform *rollout_width* simulations and compute average Q‑score."""
        lengths: List[int] = []
        successes = 0
        for _ in range(self.config.rollout_width):
            prompt = self._prompt_rollout(question, node.state)
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            with torch.no_grad():
                out = self.model.generate(input_ids=ids, **self.gen_cfg_rollout.to_dict())
            gen_ids = out[0][ids.shape[-1]:]
            lengths.append(len(gen_ids))
            generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            full_solution = node.state + generated
            ans = self._extract_answer(full_solution)
            print(f"[Rollout] Solution:\n{full_solution}\n=> Extracted answer: {ans}")
            gold = self.golden_answers.get(question)
            print("Extracted rollout_answer:", ans)
            if ans is not None and gold is not None and self._compare_answers(ans, gold):
                successes += 1

        # update cumulative MC statistics ----------------------------------
        node.correct_rollouts += successes
        node.total_rollouts += self.config.rollout_width
        mc = node.correct_rollouts / max(1, node.total_rollouts)

        # compute Q‑scores for each rollout length --------------------------
        q_scores = [
            (self.config.alpha ** (1 - mc)) * (self.config.beta ** (l / self.config.L))
            for l in lengths
        ]
        value = sum(q_scores) / len(q_scores)
        print(f"[Rollout] successes: {successes}/{self.config.rollout_width}, mc={mc:.2f}, Q={value:.3f}")
        return value

    @staticmethod
    def _compare_answers(pred: str, gold: str) -> bool: # Loose numeric match – can be improved to exact or symbolic comparison
        try:
            return float(pred) == float(gold)
        except ValueError:
            return pred.strip() == gold.strip()

    # ---------------------------------------------------------------------
    # 1‑D. Back‑propagation
    # ---------------------------------------------------------------------
    def _backprop(self, node: Node, outcome: float):
        while node is not None:
            node.n_visits += 1
            node.q_value += (outcome - node.q_value) / node.n_visits    # Incremental mean update
            # start from leaf
            # temp = node  
            # print(f"[Backprop] Node(partial=\"{temp.state}\") visits={temp.n_visits}, Q={temp.q_value:.3f}")
            node = node.parent

    # ---------------------------------------------------------------------
    # 1‑E. Public interface – run one search
    # ---------------------------------------------------------------------
    def solve(self, question: str, iterations: int = 2) -> Tuple[str, Optional[str], float]:
        """Run MCTS for *iterations* simulations and return best solution."""
        self.root = Node(state="", parent=None, prior=0.0, children={})
        for _ in trange(iterations, desc="MCTS"):
            # 1. Selection
            leaf = self._select(self.root)
            # 2. Expansion
            # self._expand(leaf, question)
            new_child = self._expand(leaf, question)
            # 3. Simulation
            # value = self._rollout_from(leaf, question)
            sim_node = new_child if new_child is not None else leaf
            value = self._rollout_from(sim_node, question)
            # 4. Back‑propagation
            self._backprop(leaf, value)

        # Choose the most visited child of root as final solution path
        if not self.root.children:
            return "", None, 0.0
        best_step, best_child = max(self.root.children.items(), key=lambda kv: kv[1].n_visits)
        
        # Optionally run one deterministic rollout from best_child to get a full solution
        prompt = self._prompt_rollout(question, best_child.state)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, **self.gen_cfg_rollout.to_dict())
        gen = self.tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
        full_solution = best_child.state + gen
        answer = self._extract_answer(full_solution)
        return full_solution, answer, best_child    # best_child.q_value

    # ---------------------------------------------------------------------
    # 2. Interface with main function
    # ---------------------------------------------------------------------
    def mcts_for_prm(self, q: str, samples: int = 1) -> Dict[str, List[Dict]]:
        """
        Runs MCTS to collect high-quality solution paths for the given question.
        Returns a dict with the question as key and a list of solution entries (with steps and rewards) as value.
        """
        results = []  # will collect solution entries for this question
        for _ in range(samples):
            s, a, node = self.solve(q)

            # Filter out solutions with incorrect final answers (if golden answer is provided)
            gold_answer = self.golden_answers.get(q, None)
            gold_answer = self._extract_answer(gold_answer)
            if gold_answer is not None:
                if a is None or not (a==gold_answer):
                    continue  # skip this path since final answer is wrong

            # Determine final solution reward based on configuration
            if hasattr(self.config, "use_mc_reward") and not self.config.use_mc_reward:
                # Use value estimate (q_val) as reward if available
                score_value = getattr(node, "q_value", None)
                if score_value is None:
                    score_value = node.correct_rollouts / max(1, node.total_rollouts)
            else:
                # Default: use Monte Carlo success rate
                score_value = node.correct_rollouts / max(1, node.total_rollouts)

            # If no gold answer, apply a quality threshold on the reward (e.g., require high success rate)
            if gold_answer is None and hasattr(self.config, "reward_threshold"):
                if score_value < self.config.reward_threshold:
                    continue  # skip low-quality path

            # Reconstruct the sequence of steps from the root to this final node
            path_nodes = []
            curr = node
            while curr.parent is not None:           # traverse back to root (excluding the root itself)
                path_nodes.append(curr)
                curr = curr.parent
            path_nodes.reverse()                    # now from first step to last step node

            # Split the solution text `s` into individual steps.
            # (Assumes each reasoning step is separated by a newline in `s`.)
            if "\n" in s:
                steps_text = [line.strip() for line in s.splitlines() if line.strip()]
                # If the first line of s was the question/prompt, remove it
                if len(steps_text) > len(path_nodes):
                    steps_text = steps_text[-len(path_nodes):]
            else:
                steps_text = [s]  # if no explicit step separation, treat the whole solution as one step

            # Collect reward for each step node (MC success or q_val as configured)
            step_rewards = []
            for nd in path_nodes:
                if hasattr(self.config, "use_mc_reward") and not self.config.use_mc_reward:
                    step_val = getattr(nd, "q_value", None)
                    if step_val is None:
                        step_val = nd.correct_rollouts / max(1, nd.total_rollouts)
                else:
                    step_val = nd.correct_rollouts / max(1, nd.total_rollouts)
                step_rewards.append(step_val)

            # Ensure the number of steps matches the number of rewards (trim if necessary)
            if len(steps_text) != len(step_rewards):
                min_len = min(len(steps_text), len(step_rewards))
                steps_text = steps_text[:min_len]
                step_rewards = step_rewards[:min_len]

            # Save this solution path entry
            results.append({
                "question": q,
                "completion": steps_text,
                "rewards": step_rewards,
                "answer": gold_answer if gold_answer is not None else a
            })

            print("MCTS for PRM data format", results)

        # Merge duplicate solution paths (average their rewards if seen multiple times)
        merged = {}
        for entry in results:
            # Use tuple of steps as a key for identity of solution path
            key = tuple(entry["completion"])
            if key in merged:
                # Already have this path: average the step-wise rewards
                old = merged[key]
                avg_rewards = [
                    (r_old + r_new) / 2.0 
                    for r_old, r_new in zip(old["rewards"], entry["rewards"])
                ]
                old["rewards"] = avg_rewards
                merged[key] = old
            else:
                merged[key] = entry

        # Return a dictionary with question as key and list of solution entries as value
        return {q: list(merged.values())}

    # -- metrics / export ----------------------------------------------
    def _collect_nodes(self): 
        stack = [self.root]
        while stack:
            n = stack.pop(); yield n; stack.extend(n.children.values())

    def get_metrics(self) -> Dict[str, float]:
        leaves = sum(len(n.children) == 0 for n in self._collect_nodes())
        return {"total_nodes": len(list(self._collect_nodes())), "leaf_nodes": leaves}

    def export_results(self, path: str):
        with open(path, "w") as f: 
            json.dump(self.get_metrics(), f, indent=2)

    def print_tree(self, node: Node, depth: int = 0):
        prefix = "    " * depth
        state_preview = node.state.replace("\n", " / ")  # 줄바꿈을 슬래시로 치환하여 한 줄로 표시
        if len(state_preview) > 60:  # 너무 길면 자르기
            state_preview = state_preview[:57] + "..."
        print(f"{prefix}- Node(depth={depth}, visits={node.n_visits}, Q={node.q_value:.2f}): {state_preview}")
        for child_step, child_node in node.children.items():
            self.print_tree(child_node, depth + 1)
