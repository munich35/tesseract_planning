/**
 * @file trajopt_planner.cpp
 * @brief Tesseract ROS Trajopt planner
 *
 * @author Levi Armstrong
 * @date April 18, 2018
 *
 * @copyright Copyright (c) 2017, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <tesseract/common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <console_bridge/console.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <trajopt_sqp/trust_region_sqp_solver.h>
#include <trajopt_sqp/trajopt_qp_problem.h>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include <tesseract_motion_planners/trajopt_ifopt/trajopt_ifopt_motion_planner.h>
#include <tesseract_motion_planners/trajopt_ifopt/trajopt_ifopt_utils.h>
#include <tesseract_motion_planners/trajopt_ifopt/profile/trajopt_ifopt_default_move_profile.h>
#include <tesseract_motion_planners/trajopt_ifopt/profile/trajopt_ifopt_default_composite_profile.h>
#include <tesseract_motion_planners/trajopt_ifopt/profile/trajopt_ifopt_osqp_solver_profile.h>
#include <tesseract_motion_planners/core/types.h>
#include <tesseract_motion_planners/planner_utils.h>

#include <tesseract/common/joint_state.h>
#include <tesseract/common/profile_dictionary.h>
#include <tesseract/kinematics/joint_group.h>
#include <tesseract/kinematics/kinematic_group.h>
#include <tesseract/environment/environment.h>
#include <tesseract_command_language/poly/move_instruction_poly.h>
#include <tesseract_command_language/poly/waypoint_poly.h>
#include <tesseract_command_language/utils.h>

// GP25 fable-constraint-spike debug hook (flag-gated, .cpp-body only, ABI-safe)
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <unistd.h>

constexpr auto SOLUTION_FOUND{ "Found valid solution" };
constexpr auto ERROR_INVALID_INPUT{ "Failed invalid input" };
constexpr auto ERROR_FAILED_TO_FIND_VALID_SOLUTION{ "Failed to find valid solution" };

namespace tesseract::motion_planners
{
TrajOptIfoptMotionPlanner::TrajOptIfoptMotionPlanner(std::string name) : MotionPlanner(std::move(name)) {}

bool TrajOptIfoptMotionPlanner::terminate()
{
  CONSOLE_BRIDGE_logWarn("Termination of ongoing optimization is not implemented yet");
  return false;
}

void TrajOptIfoptMotionPlanner::clear() {}

std::unique_ptr<MotionPlanner> TrajOptIfoptMotionPlanner::clone() const
{
  return std::make_unique<TrajOptIfoptMotionPlanner>(name_);
}

PlannerResponse TrajOptIfoptMotionPlanner::solve(const PlannerRequest& request) const
{
  PlannerResponse response;
  std::string reason;
  if (!checkRequest(request, reason))
  {
    response.successful = false;
    response.message = std::string(ERROR_INVALID_INPUT) + reason;
    return response;
  }

  // Get composite manip info
  const tesseract::common::ManipulatorInfo& composite_mi = request.instructions.getManipulatorInfo();
  if (composite_mi.empty())
    throw std::runtime_error("TrajoptIfoptMotionPlanner, manipulator info is empty!");

  // Flatten the input for planning
  auto move_instructions = request.instructions.flatten(&tesseract::command_language::moveFilter);

  // Store fixed steps
  std::vector<int> fixed_steps;

  // ----------------
  // Translate TCL for MoveInstructions
  // ----------------
  // Transform plan instructions into trajopt cost and constraints
  std::vector<std::unique_ptr<trajopt_ifopt::Node>> nodes;
  std::vector<TrajOptIfoptWaypointInfo> wp_infos;
  nodes.reserve(move_instructions.size());
  wp_infos.reserve(move_instructions.size());
  for (int i = 0; i < move_instructions.size(); ++i)
  {
    const auto& move_instruction =
        move_instructions[static_cast<std::size_t>(i)].get().as<tesseract::command_language::MoveInstructionPoly>();

    // Get Plan Profile
    // GP25 debug: default is a named local so we can detect dictionary-miss fallback by pointer identity
    auto fable_default_move_profile = std::make_shared<TrajOptIfoptDefaultMoveProfile>();
    TrajOptIfoptMoveProfile::ConstPtr cur_move_profile = request.profiles->getProfile<TrajOptIfoptMoveProfile>(
        name_, move_instruction.getProfile(name_), fable_default_move_profile);
    if (!cur_move_profile)
      throw std::runtime_error("TrajOptIfoptMotionPlanner: Invalid profile");

    // GP25 fable-constraint-spike: log per-move profile-key resolution (flag-gated, first 70 lines)
    {
      static const bool fable_dbg_on = []() {
        const char* home = std::getenv("HOME");
        const std::string flag_path = (home != nullptr ? std::string(home) : std::string()) +
                                      "/.config/gp25_trajopt_debug.on";
        return std::ifstream(flag_path).good();
      }();
      static std::atomic<int> fable_dbg_lines{ 0 };
      if (fable_dbg_on && fable_dbg_lines.fetch_add(1) < 70)
      {
        std::ofstream fable_log("/tmp/fable_keylog_" + std::to_string(::getpid()) + ".txt", std::ios::app);
        if (fable_log)
        {
          const auto& fable_wp = move_instruction.getWaypoint();
          const char* fable_wp_type = fable_wp.isStateWaypoint()     ? "state" :
                                      fable_wp.isCartesianWaypoint() ? "cartesian" :
                                      fable_wp.isJointWaypoint()     ? "joint" :
                                                                       "other";
          std::ostringstream fable_coeff;
          const auto* fable_dmp = dynamic_cast<const TrajOptIfoptDefaultMoveProfile*>(cur_move_profile.get());
          if (fable_dmp != nullptr)
            fable_coeff << fable_dmp->cartesian_constraint_config.coeff.transpose();
          else
            fable_coeff << "<resolved profile is not TrajOptIfoptDefaultMoveProfile: "
                        << typeid(*cur_move_profile).name() << ">";
          const bool fable_fallback =
              (cur_move_profile.get() ==
               static_cast<const TrajOptIfoptMoveProfile*>(fable_default_move_profile.get()));
          fable_log << "i=" << i << " ns='" << name_ << "'"
                    << " raw_profile='" << move_instruction.getProfile(name_) << "'"
                    << " wp=" << fable_wp_type
                    << " cart_coeff=[" << fable_coeff.str() << "]"
                    << " fallback=" << (fable_fallback ? 1 : 0) << "\n";
        }
      }
    }

    // Create waypoint info
    TrajOptIfoptWaypointInfo wp_info = cur_move_profile->create(move_instruction, composite_mi, request.env, i);

    // Update fixed steps
    if (wp_info.fixed)
      fixed_steps.push_back(i);

    // Add Node
    nodes.emplace_back(std::move(wp_info.node));
    wp_infos.emplace_back(std::move(wp_info));
  }

  auto variables = std::make_shared<trajopt_ifopt::NodesVariables>("joint_trajectory", std::move(nodes));
  auto nlp = std::make_shared<trajopt_sqp::TrajOptQPProblem>(variables);

  for (const auto& wp_info : wp_infos)
  {
    // Add Waypoint Cost and Constraints
    for (const auto& cnt : wp_info.term_infos.constraints)
      nlp->addConstraintSet(cnt);

    for (const auto& absolute_cost : wp_info.term_infos.absolute_costs)
      nlp->addCostSet(absolute_cost, trajopt_sqp::CostPenaltyType::kAbsolute);

    for (const auto& squared_cost : wp_info.term_infos.squared_costs)
      nlp->addCostSet(squared_cost, trajopt_sqp::CostPenaltyType::kSquared);

    for (const auto& hinge_cost : wp_info.term_infos.hinge_costs)
      nlp->addCostSet(hinge_cost, trajopt_sqp::CostPenaltyType::kHinge);
  }

  // ----------------
  // Translate TCL for CompositeInstructions
  // ----------------
  TrajOptIfoptCompositeProfile::ConstPtr cur_composite_profile =
      request.profiles->getProfile<TrajOptIfoptCompositeProfile>(
          name_, request.instructions.getProfile(name_), std::make_shared<TrajOptIfoptDefaultCompositeProfile>());

  if (!cur_composite_profile)
    throw std::runtime_error("TrajOptIfoptMotionPlanner: Invalid profile");

  TrajOptIfoptTermInfos term_infos =
      cur_composite_profile->create(composite_mi, request.env, variables->getNodes(), fixed_steps);

  // Add Waypoint Cost and Constraints
  for (const auto& cnt : term_infos.constraints)
    nlp->addConstraintSet(cnt);

  for (const auto& absolute_cost : term_infos.absolute_costs)
    nlp->addCostSet(absolute_cost, trajopt_sqp::CostPenaltyType::kAbsolute);

  for (const auto& squared_cost : term_infos.squared_costs)
    nlp->addCostSet(squared_cost, trajopt_sqp::CostPenaltyType::kSquared);

  for (const auto& hinge_cost : term_infos.hinge_costs)
    nlp->addCostSet(hinge_cost, trajopt_sqp::CostPenaltyType::kHinge);

  // Setup
  nlp->setup();

  // Create Solver
  TrajOptIfoptSolverProfile::ConstPtr solver_profile = request.profiles->getProfile<TrajOptIfoptSolverProfile>(
      name_, request.instructions.getProfile(name_), std::make_shared<TrajOptIfoptOSQPSolverProfile>());

  if (!solver_profile)
    throw std::runtime_error("TrajOptIfoptMotionPlanner: Invalid profile");

  std::unique_ptr<trajopt_sqp::TrustRegionSQPSolver> solver = solver_profile->create(request.verbose);

  // Solver
  solver->solve(nlp);

  // Check success
  if (solver->getStatus() != trajopt_sqp::SQPStatus::kConverged)
  {
    response.successful = false;
    response.message =
        std::string(ERROR_FAILED_TO_FIND_VALID_SOLUTION).append(": ").append(toString(solver->getStatus()));
  }
  else
  {
    response.successful = true;
    response.message = SOLUTION_FOUND;
  }

  // [fable P0a 2026-07-12] PRIVATE JointGroup per solve: Environment::clone() SHARES the
  // joint_group_cache (environment.cpp:460), so the cached JointGroup (one mutex around the
  // non-thread-safe KDL state solver) is contended by every in-process consumer — during a GP25
  // pack each FK/Jacobian serialized, inflating one SQP iteration from ~0.03 s to 60-77 s.
  // Upstream fixed this pattern for classic trajopt only (63ab0a34); trajopt_ifopt needs it too.
  auto manip = std::make_shared<tesseract::kinematics::JointGroup>(*request.env->getJointGroup(composite_mi.manipulator));
  const std::vector<std::string> joint_names = manip->getJointNames();
  const Eigen::MatrixX2d joint_limits = manip->getLimits().joint_limits;

  // Get the results - This can likely be simplified if we get rid of the traj array
  Eigen::VectorXd x = nlp->getVariableValues();
  Eigen::Map<tesseract::common::TrajArray> traj(
      x.data(), static_cast<Eigen::Index>(move_instructions.size()), manip->numJoints());

  // Enforce limits
  for (Eigen::Index i = 0; i < traj.rows(); i++)
  {
    assert(tesseract::common::satisfiesLimits<double>(traj.row(i), joint_limits, 1e-4));
    tesseract::common::enforceLimits<double>(traj.row(i), joint_limits);
  }

  // Flatten the results to make them easier to process
  response.results = request.instructions;
  auto results_instructions = response.results.flatten(&tesseract::command_language::moveFilter);
  assert(results_instructions.size() == traj.rows());
  for (std::size_t idx = 0; idx < results_instructions.size(); idx++)
  {
    auto& move_instruction = results_instructions.at(idx).get().as<tesseract::command_language::MoveInstructionPoly>();
    assignSolution(
        move_instruction, joint_names, traj.row(static_cast<Eigen::Index>(idx)), request.format_result_as_input);
  }

  return response;
}

}  // namespace tesseract::motion_planners
