from __future__ import annotations

import json
from pathlib import Path

SNIPPETS: dict[str, str] = {
    "d01": '''
def preprocess_input(raw_text):
    """Strip surrounding whitespace before handing off to the main routine."""
    cleaned = raw_text.strip()
    return main_handler(cleaned)
''',
    "d02": '''
def check_locale_prefix(value):
    """Return True when the value begins with a supported locale tag."""
    for prefix in SUPPORTED_LOCALES:
        if value.startswith(prefix):
            return True
    return False
''',
    "d03": '''
def render_report(rows):
    total = 0
    for row in rows:
        total += row.amount
    return "Total: %d" % total
''',
    "d04": '''
def is_palindrome(s):
    """Check whether a string reads the same forwards and backwards."""
    cleaned = "".join(ch.lower() for ch in s if ch.isalnum())
    return cleaned == cleaned[::-1]
''',
    "d05": '''
def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for item in strs[1:]:
        while not item.startswith(prefix):
            prefix = prefix[:-1]
    return prefix
''',
    "d06": '''
def merge_sort(arr):
    """Sort a list of integers in ascending order using merge sort."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)
''',
    "d07": '''
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    smaller = [x for x in arr if x < pivot]
    equal = [x for x in arr if x == pivot]
    larger = [x for x in arr if x > pivot]
    return quick_sort(smaller) + equal + quick_sort(larger)
''',
    "d08": '''
def sort_by_frequency(items):
    """Order elements by how often they occur, most frequent first."""
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return sorted(items, key=lambda x: (-counts[x], x))
''',
    "d09": '''
def binary_search(arr, target):
    """Find the index of target in a sorted array, or -1 if absent."""
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
    "d10": '''
def linear_scan(arr, target):
    for index, value in enumerate(arr):
        if value == target:
            return index
    return -1
''',
    "d11": '''
def gcd(a, b):
    """Greatest common divisor by the Euclidean algorithm."""
    while b:
        a, b = b, a % b
    return a
''',
    "d12": '''
def lcm(a, b):
    return a * b // gcd(a, b)
''',
    "d13": '''
def sieve_of_eratosthenes(n):
    """Return every prime number up to n."""
    flags = [True] * (n + 1)
    flags[0] = flags[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if flags[i]:
            for j in range(i * i, n + 1, i):
                flags[j] = False
    return [i for i, ok in enumerate(flags) if ok]
''',
    "d14": '''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
''',
    "d15": '''
def fibonacci(n):
    """Compute the nth Fibonacci number iteratively."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
    "d16": '''
def climb_stairs(n):
    """Count the distinct ways to climb n stairs taking 1 or 2 steps."""
    first, second = 1, 1
    for _ in range(n - 1):
        first, second = second, first + second
    return second
''',
    "d17": '''
def read_and_solve():
    n = int(input())
    values = list(map(int, input().split()))
    print(sum(values) // n)
''',
    "d18": '''
def write_results(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(str(row) + "\\n")
''',
    "d19": '''
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
''',
    "d20": '''
class LinkedList:
    """Singly linked list with push and reverse operations."""
    def __init__(self):
        self.head = None

    def push(self, value):
        node = Node(value)
        node.next = self.head
        self.head = node

    def reverse(self):
        prev = None
        current = self.head
        while current:
            nxt = current.next
            current.next = prev
            prev = current
            current = nxt
        self.head = prev
''',
    "d21": '''
def max_subarray_sum(nums):
    """Largest sum of any contiguous subarray (Kadane's algorithm)."""
    best = current = nums[0]
    for value in nums[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best
''',
    "d22": '''
def two_sum(nums, target):
    """Return indices of the two numbers adding up to target."""
    seen = {}
    for index, value in enumerate(nums):
        if target - value in seen:
            return [seen[target - value], index]
        seen[value] = index
    return []
''',
    "d23": '''
def normalize_whitespace(text):
    """Collapse runs of whitespace into single spaces and trim the ends."""
    return " ".join(text.split())
''',
    "d24": '''
def sanitize_filename(name):
    bad = "<>:/\\\\|?*"
    for ch in bad:
        name = name.replace(ch, "_")
    return name.strip()
''',
    "d25": '''
def bfs_shortest_path(graph, start, goal):
    """Breadth-first search for the shortest path between two nodes."""
    from collections import deque
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for neighbour in graph.get(node, []):
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, path + [neighbour]))
    return []
''',
    "d26": '''
def dfs_visit(graph, node, seen=None):
    if seen is None:
        seen = set()
    seen.add(node)
    for neighbour in graph.get(node, []):
        if neighbour not in seen:
            dfs_visit(graph, neighbour, seen)
    return seen
''',
    "d27": '''
def count_vowels(text):
    return sum(1 for ch in text.lower() if ch in "aeiou")
''',
    "d28": '''
def reverse_words(sentence):
    """Reverse the order of words while keeping each word intact."""
    return " ".join(sentence.split()[::-1])
''',
    "d29": '''
def matrix_transpose(matrix):
    return [list(row) for row in zip(*matrix)]
''',
    "d30": '''
def validate_email(address):
    """Check that an address contains exactly one @ and a dot after it."""
    if address.count("@") != 1:
        return False
    local, _, domain = address.partition("@")
    return bool(local) and "." in domain
''',
}

QUERIES: dict[str, tuple[str, list[str]]] = {
    "q01": ("How is the input preprocessed before going to the main function?", ["d01"]),
    "q02": ("Check whether a word reads the same in both directions", ["d04"]),
    "q03": ("Sort a list of numbers into increasing order", ["d06", "d07"]),
    "q04": ("Find an element quickly in an already sorted list", ["d09"]),
    "q05": ("Compute the greatest common divisor of two integers", ["d11"]),
    "q06": ("List every prime number below a given limit", ["d13"]),
    "q07": ("Work out the nth term of the Fibonacci sequence", ["d15"]),
    "q08": ("Read numbers from standard input and print their average", ["d17"]),
    "q09": ("Reverse a singly linked list in place", ["d20"]),
    "q10": ("Largest sum of a contiguous run of numbers in an array", ["d21"]),
    "q11": ("Find two entries that add up to a given total", ["d22"]),
    "q12": ("Remove extra spaces from a piece of text", ["d23"]),
    "q13": ("Shortest route between two points in a graph", ["d25"]),
    "q14": ("Order items by how frequently they appear", ["d08"]),
    "q15": ("Validate that an email address is well formed", ["d30"]),
    "q16": ("Flip the order of words in a sentence", ["d28"]),
    "q17": ("Get rid of leading and trailing blanks in a piece of text", ["d23"]),
    "q18": ("Figure out whether a number has no divisors besides one and itself",
            ["d14"]),
    "q19": ("Turn a grid so that its rows become columns", ["d29"]),
    "q20": ("How many times do a, e, i, o and u show up in a word", ["d27"]),
    "q21": ("Visit every point you can reach starting from one place", ["d26"]),
    "q22": ("Make a file name safe by swapping out illegal characters", ["d24"]),
    "q23": ("Longest starting section shared by a group of words", ["d05"]),
    "q24": ("Count the different routes up a staircase one or two at a time",
            ["d16"]),
    "q25": ("Smallest number that both values divide into evenly", ["d12"]),
    "q26": ("Save each entry onto its own line of a document", ["d18"]),
}


def main() -> None:
    corpus = {key: value.strip("\n") for key, value in SNIPPETS.items()}
    queries = {qid: text for qid, (text, _) in QUERIES.items()}
    qrels = {qid: {doc: 1 for doc in docs} for qid, (_, docs) in QUERIES.items()}

    missing = {doc for rels in qrels.values() for doc in rels} - set(corpus)
    if missing:
        raise SystemExit(f"qrels reference unknown documents: {sorted(missing)}")

    out = Path(__file__).resolve().parent / "sample_dataset.json"
    out.write_text(json.dumps(
        {"corpus": corpus, "queries": queries, "qrels": qrels},
        indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  {len(corpus)} snippets, {len(queries)} queries, "
          f"{sum(len(r) for r in qrels.values())} judgements")


if __name__ == "__main__":
    main()
