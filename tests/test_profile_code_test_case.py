"""Profile 驱动代码与测试案例的离线验证。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from examples.profile_code_test_case import SOURCE_FILE, TEST_FILE, run_case


# 案例应分别通过代码和测试角色上下文写入文件，并让宿主 pytest 成功验证它们。
class ProfileCodeTestCaseTests(unittest.TestCase):
    def test_run_case_writes_source_and_test_then_runs_pytest(self: "ProfileCodeTestCaseTests") -> None:
        with TemporaryDirectory() as temporary_directory:
            workspace_root = Path(temporary_directory)

            result = run_case(workspace_root)

            self.assertEqual(result.exit_code, 0, result.stdout + result.stderr)
            self.assertFalse(result.timed_out)
            self.assertTrue((workspace_root / SOURCE_FILE).is_file())
            self.assertTrue((workspace_root / TEST_FILE).is_file())
