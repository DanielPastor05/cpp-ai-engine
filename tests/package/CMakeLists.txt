# A consumer of the *installed* engine, and deliberately not part of the main
# build. It is configured on its own, against an install prefix, with no
# knowledge of the source tree -- because that is the only arrangement that
# exercises the exported package instead of the targets that already exist in
# the parent scope.
#
# This is a real test and it found a real bug the first time it ran: `engine`
# links Threads::Threads as PUBLIC, the generated config file never re-found
# Threads, and every consumer outside this repository failed at configure time
# with "the link interface of target engine::engine contains Threads::Threads
# but the target was not found". The install rules had been in place for weeks.
# Nothing inside the repository could have caught it.
cmake_minimum_required(VERSION 3.18)
project(engine_package_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# REQUIRED, and no CONFIG hint: if the package is not found, or is found and is
# broken, this fails here rather than producing a build that quietly did nothing.
find_package(cpp_ai_engine REQUIRED)

add_executable(package_test main.cpp)

# The namespaced alias, not the bare `engine` -- a consumer only ever has the
# alias, and using the plain name here would test something the outside world
# cannot do.
target_link_libraries(package_test PRIVATE engine::engine)
