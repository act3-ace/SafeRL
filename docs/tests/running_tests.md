# How To Run Tests

---

## Quickstart:
From the **have-deepsky/tests** directory, run the following command:

`pytest`

This will launch an automatic collection and execution of every test function found within our testing suite.

## Test Organization
Our testing suite leverages [Pytest](https://docs.pytest.org/en/6.2.x/).
Test functions are organized into three general categories: unit tests, integration tests, and system tests.
To save time and integrate into our development cycle, each category of tests can be run independently.
To do so, navigate to the **have-deepsky/tests** directory and simply pass the desired category marker to pytest on the command line.

To run only unit tests:

`pytest -m unit_test`

Integration tests:

`pytest -m integration_test`

System tests:

`pytest -m system_test`
