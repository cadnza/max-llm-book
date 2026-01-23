# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test that tutorial resources (checks and solutions) are valid Python."""


class TestCheckModules:
    """Test that check modules can be imported."""

    def test_import_check_modules(self) -> None:
        """Verify all check modules are valid Python."""
        # Import each check module to ensure they're syntactically valid
        # ruff: noqa: F401
        import check_step_01
        import check_step_02
        import check_step_03
        import check_step_04
        import check_step_05
        import check_step_06
        import check_step_07
        import check_step_08
        import check_step_09
        import check_step_10
        import check_step_11

        # If we get here, all imports succeeded
        assert True


class TestSolutionModules:
    """Test that solution modules can be imported."""

    def test_import_solution_modules(self) -> None:
        """Verify all solution modules are valid Python."""
        # Import each solution module to ensure they're syntactically valid
        # ruff: noqa: F401
        import solution_01
        import solution_02
        import solution_03
        import solution_04
        import solution_05
        import solution_06
        import solution_07
        import solution_08
        import solution_09
        import solution_10
        import solution_11

        # If we get here, all imports succeeded
        assert True
