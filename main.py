import json
import time
from typing import Dict, List, Optional, Tuple


EPSILON = 1e-9
REPEAT_COUNT = 10


class MiniNPUSimulator:
    """MAC(Multiply-Accumulate) 연산 기반 패턴 판별 시뮬레이터"""

    def __init__(self, data_path: str = "data.json") -> None:
        self.data_path = data_path
        self.filters: Dict[int, Dict[str, List[List[float]]]] = {}
        self.raw_patterns: Dict[str, dict] = {}

    @staticmethod
    def normalize_label(raw_label: str) -> Optional[str]:
        """다양한 입력 라벨('+', 'cross', 'x' 등)을 표준 라벨('Cross', 'X')로 정규화"""
        if not isinstance(raw_label, str):
            return None
        cleaned = raw_label.strip().lower()
        if cleaned in ("+", "cross"):
            return "Cross"
        if cleaned == "x":
            return "X"
        return None

    @staticmethod
    def validate_square_matrix(matrix: object, expected_size: int) -> Tuple[bool, str]:
        """2차원 리스트가 expected_size x expected_size 숫자 행렬인지 검증"""
        if not isinstance(matrix, list) or len(matrix) != expected_size:
            actual_rows = len(matrix) if isinstance(matrix, list) else "N/A"
            return False, f"행 개수 불일치 (기대: {expected_size}, 실제: {actual_rows})"

        for row_idx, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != expected_size:
                actual_cols = len(row) if isinstance(row, list) else "N/A"
                return False, f"{row_idx + 1}행 열 개수 불일치 (기대: {expected_size}, 실제: {actual_cols})"
            for col_idx, val in enumerate(row):
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    return False, f"({row_idx + 1}, {col_idx + 1}) 위치 값이 숫자가 아님"

        return True, ""

    @staticmethod
    def mac_compute(pattern: List[List[float]], flt: List[List[float]]) -> float:
        """외부 라이브러리 없이 이중 반복문으로 MAC(곱셈 후 누적합) 연산 수행"""
        size = len(pattern)
        total = 0.0
        for i in range(size):
            for j in range(size):
                total += pattern[i][j] * flt[i][j]
        return total

    def judge_scores(
        self,
        score_a: float,
        score_b: float,
        label_a: str = "A",
        label_b: str = "B",
        tie_label: str = "UNDECIDED",
    ) -> str:
        """허용오차(epsilon = 1e-9) 기반 점수 비교 및 판정"""
        if abs(score_a - score_b) < EPSILON:
            return tie_label
        return label_a if score_a > score_b else label_b

    def measure_mac_time_ms(
        self,
        pattern: List[List[float]],
        flt_a: List[List[float]],
        flt_b: List[List[float]],
        repeats: int = REPEAT_COUNT,
    ) -> float:
        """I/O를 제외하고 순수 MAC 연산 함수 호출 구간만 repeats회 반복 측정 후 평균 시간(ms) 반환"""
        start = time.perf_counter()
        for _ in range(repeats):
            self.mac_compute(pattern, flt_a)
            self.mac_compute(pattern, flt_b)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return elapsed_ms / repeats

    @staticmethod
    def print_performance_table(records: List[Tuple[int, float]]) -> None:
        """크기(N×N), 평균 시간(ms), 연산 횟수(N²) 성능 분석 표 출력"""
        print("\n#----------------------------------------")
        print(f"# [성능 분석] (평균/{REPEAT_COUNT}회 측정, I/O 제외)")
        print("#----------------------------------------")
        print(f"{'크기':<10} {'평균 시간(ms)':<16} {'연산 횟수(N²)'}")
        print("-" * 42)
        for size, avg_ms in records:
            ops = size * size
            print(f"{f'{size}×{size}':<10} {avg_ms:<16.4f} {ops}")

    def read_3x3_matrix(self, title: str) -> List[List[float]]:
        """모드 1에서 3×3 행렬을 한 줄씩(공백 구분) 입력받고 행/열 및 숫자 파싱 검증"""
        while True:
            print(f"\n{title} (3줄 입력, 공백 구분)")
            matrix: List[List[float]] = []
            valid = True
            for row_num in range(1, 4):
                raw_line = input(f"  {row_num}행: ").strip()
                tokens = raw_line.split()
                if len(tokens) != 3:
                    print("입력 형식 오류: 각 줄에 3개의 숫자를 공백으로 구분해 입력하세요.")
                    valid = False
                    break
                try:
                    row = [float(tok) for tok in tokens]
                except ValueError:
                    print("입력 형식 오류: 숫자(정수 또는 실수)만 입력할 수 있습니다.")
                    valid = False
                    break
                matrix.append(row)

            if valid and len(matrix) == 3:
                return matrix

    def run_user_input_mode(self) -> None:
        """모드 1: 사용자 입력(3×3) 실행 흐름"""
        print("\n#----------------------------------------")
        print("# [1] 필터 입력")
        print("#----------------------------------------")
        filter_a = self.read_3x3_matrix("필터 A")
        filter_b = self.read_3x3_matrix("필터 B")
        print("✓ 필터 A, B (3×3) 저장 완료")

        print("\n#----------------------------------------")
        print("# [2] 패턴 입력")
        print("#----------------------------------------")
        pattern = self.read_3x3_matrix("패턴")
        print("✓ 입력 패턴 (3×3) 저장 완료")

        score_a = self.mac_compute(pattern, filter_a)
        score_b = self.mac_compute(pattern, filter_b)
        avg_ms = self.measure_mac_time_ms(pattern, filter_a, filter_b, REPEAT_COUNT)
        verdict = self.judge_scores(
            score_a,
            score_b,
            label_a="A",
            label_b="B",
            tie_label="판정 불가 (|A-B| < 1e-9)",
        )

        print("\n#----------------------------------------")
        print("# [3] MAC 결과")
        print("#----------------------------------------")
        print(f"A 점수: {score_a:.16f}" if verdict.startswith("판정 불가") else f"A 점수: {score_a}")
        print(f"B 점수: {score_b:.16f}" if verdict.startswith("판정 불가") else f"B 점수: {score_b}")
        print(f"연산 시간(평균/{REPEAT_COUNT}회): {avg_ms:.4f} ms")
        print(f"판정: {verdict}")

        self.print_performance_table([(3, avg_ms)])

    def load_and_normalize_json(self) -> bool:
        """data.json 로드 및 필터 라벨 정규화"""
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"오류: '{self.data_path}' 파일을 찾을 수 없습니다.")
            return False
        except json.JSONDecodeError as e:
            print(f"오류: '{self.data_path}' JSON 파싱 실패 ({e})")
            return False

        raw_filters = data.get("filters")
        raw_patterns = data.get("patterns")
        if not isinstance(raw_filters, dict) or not isinstance(raw_patterns, dict):
            print("오류: data.json 최상위 스키마('filters', 'patterns')가 올바르지 않습니다.")
            return False

        self.filters.clear()
        self.raw_patterns = raw_patterns

        print("\n#----------------------------------------")
        print("# [1] 필터 로드")
        print("#----------------------------------------")
        for size_key, filter_group in raw_filters.items():
            if not size_key.startswith("size_") or not isinstance(filter_group, dict):
                print(f"✗ 잘못된 필터 키 형식 무시: {size_key}")
                continue
            try:
                size_n = int(size_key.split("_")[1])
            except (IndexError, ValueError):
                print(f"✗ 크기 파싱 실패 필터 무시: {size_key}")
                continue

            normalized_group: Dict[str, List[List[float]]] = {}
            for raw_label, matrix in filter_group.items():
                std_label = self.normalize_label(raw_label)
                if std_label is None:
                    print(f"✗ {size_key}: 알 수 없는 필터 라벨({raw_label})")
                    continue
                ok, reason = self.validate_square_matrix(matrix, size_n)
                if not ok:
                    print(f"✗ {size_key} ({std_label}) 스키마 오류: {reason}")
                    continue
                normalized_group[std_label] = [[float(v) for v in row] for row in matrix]

            self.filters[size_n] = normalized_group
            loaded_labels = ", ".join(normalized_group.keys())
            print(f"✓ {size_key} 필터 로드 완료 ({loaded_labels})")

        return True

    @staticmethod
    def parse_pattern_size(pattern_key: str) -> Optional[int]:
        """'size_{N}_{idx}' 형식의 패턴 키에서 N 추출"""
        parts = pattern_key.split("_")
        if len(parts) < 3 or parts[0] != "size":
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    def run_json_mode(self) -> None:
        """모드 2: data.json 분석 실행 흐름"""
        if not self.load_and_normalize_json():
            return

        print("\n#----------------------------------------")
        print("# [2] 패턴 분석 (라벨 정규화 및 스키마 검증 적용)")
        print("#----------------------------------------")

        total_count = 0
        pass_count = 0
        fail_cases: List[Tuple[str, str]] = []
        sample_by_size: Dict[int, List[List[float]]] = {}

        for pattern_key, item in self.raw_patterns.items():
            total_count += 1
            print(f"-- {pattern_key} --")

            if not isinstance(item, dict):
                reason = "스키마 오류: 패턴 항목이 딕셔너리 형태가 아님"
                print(f"판정: FAIL | 사유: {reason}")
                fail_cases.append((pattern_key, reason))
                continue

            size_n = self.parse_pattern_size(pattern_key)
            if size_n is None:
                reason = f"키 규칙 오류: '{pattern_key}'에서 크기 N 추출 실패"
                print(f"판정: FAIL | 사유: {reason}")
                fail_cases.append((pattern_key, reason))
                continue

            if size_n not in self.filters or "Cross" not in self.filters[size_n] or "X" not in self.filters[size_n]:
                reason = f"필터 누락: size_{size_n}의 Cross/X 필터가 존재하지 않음"
                print(f"판정: FAIL | 사유: {reason}")
                fail_cases.append((pattern_key, reason))
                continue

            raw_expected = item.get("expected")
            expected_label = self.normalize_label(raw_expected)
            if expected_label is None:
                reason = f"라벨 오류: expected 값('{raw_expected}') 정규화 불가"
                print(f"판정: FAIL | 사유: {reason}")
                fail_cases.append((pattern_key, reason))
                continue

            pattern_matrix = item.get("input")
            ok, matrix_reason = self.validate_square_matrix(pattern_matrix, size_n)
            if not ok:
                reason = f"크기/스키마 불일치 ({matrix_reason})"
                print(f"판정: FAIL | 사유: {reason}")
                fail_cases.append((pattern_key, reason))
                continue

            clean_pattern = [[float(v) for v in row] for row in pattern_matrix]
            if size_n not in sample_by_size:
                sample_by_size[size_n] = clean_pattern

            cross_filter = self.filters[size_n]["Cross"]
            x_filter = self.filters[size_n]["X"]

            cross_score = self.mac_compute(clean_pattern, cross_filter)
            x_score = self.mac_compute(clean_pattern, x_filter)
            verdict = self.judge_scores(
                cross_score,
                x_score,
                label_a="Cross",
                label_b="X",
                tie_label="UNDECIDED",
            )

            if verdict == "UNDECIDED":
                print(f"Cross 점수: {cross_score:.16f}")
                print(f"X 점수: {x_score:.16f}")
                print(f"판정: {verdict} | expected: {expected_label} | FAIL (동점 규칙)")
                fail_cases.append((pattern_key, "동점(UNDECIDED) 처리 규칙(|Cross - X| < 1e-9)에 따라 FAIL"))
            elif verdict == expected_label:
                pass_count += 1
                print(f"Cross 점수: {cross_score:.4f}")
                print(f"X 점수: {x_score:.4f}")
                print(f"판정: {verdict} | expected: {expected_label} | PASS")
            else:
                print(f"Cross 점수: {cross_score:.4f}")
                print(f"X 점수: {x_score:.4f}")
                print(f"판정: {verdict} | expected: {expected_label} | FAIL (예측 불일치)")
                fail_cases.append((pattern_key, f"판정 불일치 (예측: {verdict}, 정답: {expected_label})"))

        # 성능 분석 (3x3 포함, 5x5 / 13x13 / 25x25)
        perf_records: List[Tuple[int, float]] = []

        # 3x3 기본 벤치마크 데이터 (I/O 제외 순수 연산 측정)
        base_3x3_cross = [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
        base_3x3_x = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]
        perf_records.append((3, self.measure_mac_time_ms(base_3x3_cross, base_3x3_cross, base_3x3_x, REPEAT_COUNT)))

        for size_n in sorted(self.filters.keys()):
            if size_n in sample_by_size and "Cross" in self.filters[size_n] and "X" in self.filters[size_n]:
                avg_ms = self.measure_mac_time_ms(
                    sample_by_size[size_n],
                    self.filters[size_n]["Cross"],
                    self.filters[size_n]["X"],
                    REPEAT_COUNT,
                )
                perf_records.append((size_n, avg_ms))

        self.print_performance_table(perf_records)

        # 결과 요약 출력
        fail_count = len(fail_cases)
        print("\n#----------------------------------------")
        print("# [4] 결과 요약")
        print("#----------------------------------------")
        print(f"총 테스트: {total_count}개")
        print(f"통과: {pass_count}개")
        print(f"실패: {fail_count}개")

        if fail_cases:
            print("\n실패 케이스:")
            for case_id, reason in fail_cases:
                print(f"- {case_id}: {reason}")

    def run(self) -> None:
        """메인 실행 진입점"""
        print("=== Mini NPU Simulator ===")
        print("\n[모드 선택]")
        print("1. 사용자 입력 (3x3)")
        print("2. data.json 분석")

        while True:
            choice = input("선택: ").strip()
            if choice == "1":
                self.run_user_input_mode()
                break
            elif choice == "2":
                self.run_json_mode()
                break
            else:
                print("잘못된 입력입니다. 1 또는 2를 입력하세요.")


if __name__ == "__main__":
    simulator = MiniNPUSimulator("data.json")
    simulator.run()