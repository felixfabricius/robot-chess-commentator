"""Camera frame -> undistorted -> rectified board -> 64 masked square crops -> a board reading.

Nothing is imported here on purpose. `board_estimator` pulls in torch and anthropic, and several
tests import `calibration` on machines that have neither those nor the `robot` dependency group --
`test_calibration` asserts exactly that. Re-exporting anything would drag the heavy modules into
every one of those imports.
"""
