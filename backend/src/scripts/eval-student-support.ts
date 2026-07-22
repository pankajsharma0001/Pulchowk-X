import evalSetData from "../data/student_support_eval_set.json" with {
  type: "json",
};
import {
  resolveStudentConciergeQuery,
  type ConciergeAction,
  type ConciergeIntent,
} from "../services/chatbot-concierge.service.js";

interface EvalGroup {
  category: string;
  queries: string[];
}

interface EvalSet {
  version: string;
  language: string;
  groups: EvalGroup[];
}

interface CategoryExpectation {
  intents: ConciergeIntent[];
  actions: ConciergeAction[];
}

interface QueryExpectation {
  intents: ConciergeIntent[];
  actions: ConciergeAction[];
}

const CATEGORY_EXPECTATIONS: Record<string, CategoryExpectation> = {
  admissions_onboarding: {
    intents: ["process_howto", "office_lookup", "escalation", "general_campus", "unknown"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
  registration_and_records: {
    intents: ["process_howto", "office_lookup", "policy_query", "location_lookup"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
  library: {
    intents: ["policy_query", "service_lookup", "location_lookup"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
  hostel_and_food: {
    intents: ["service_lookup", "office_lookup", "location_lookup"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
  student_services: {
    intents: ["service_lookup", "office_lookup", "location_lookup"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
  notices_and_deadlines: {
    intents: ["deadline_query", "office_lookup", "policy_query", "notice_query", "escalation"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
  student_organizations: {
    intents: ["location_lookup", "office_lookup", "service_lookup", "process_howto"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
  navigation_and_route: {
    intents: ["route_navigation", "location_lookup"],
    actions: ["show_route", "show_location", "show_multiple_locations"],
  },
  emergency_and_escalation: {
    intents: ["escalation", "office_lookup", "location_lookup", "unknown"],
    actions: ["show_location", "show_multiple_locations", "text_answer"],
  },
};

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, " ").trim();
}

function looksLikeRouteQuery(query: string): boolean {
  const q = normalize(query);
  return (
    (q.includes("from ") && q.includes(" to ")) ||
    (q.includes("between ") && q.includes(" and ")) ||
    q.includes("directions") ||
    q.includes("navigate") ||
    q.includes("show route")
  );
}

function getExpectedForQuery(category: string, query: string): QueryExpectation {
  const base = CATEGORY_EXPECTATIONS[category] ?? {
    intents: ["unknown"],
    actions: ["show_location", "show_multiple_locations"],
  };

  if (category === "navigation_and_route" && looksLikeRouteQuery(query)) {
    return {
      intents: ["route_navigation"],
      actions: ["show_route"],
    };
  }

  return base;
}

function formatList(values: string[]): string {
  return values.join(", ");
}

async function run(): Promise<void> {
  const evalSet = evalSetData as EvalSet;

  let totalQueries = 0;
  let passedIntentChecks = 0;
  let passedActionChecks = 0;
  let fullPasses = 0;
  const failures: string[] = [];

  console.log(
    `Running student support eval set v${evalSet.version} (${evalSet.language})`,
  );
  console.log("=".repeat(72));

  for (const group of evalSet.groups) {
    console.log(`\n[Category] ${group.category}`);
    for (const query of group.queries) {
      totalQueries += 1;
      const expected = getExpectedForQuery(group.category, query);
      const result = await resolveStudentConciergeQuery(query, {
        allowLlm: false,
      });

      const intentPass = expected.intents.includes(result.intent);
      const actionPass = expected.actions.includes(result.action);
      const fullPass = intentPass && actionPass;

      if (intentPass) passedIntentChecks += 1;
      if (actionPass) passedActionChecks += 1;
      if (fullPass) fullPasses += 1;

      const status = fullPass ? "PASS" : "FAIL";
      console.log(
        `- [${status}] "${query}"\n` +
          `    intent: expected(${formatList(expected.intents)}) got(${result.intent}) | ` +
          `action: expected(${formatList(expected.actions)}) got(${result.action})`,
      );

      if (!fullPass) {
        failures.push(
          `[${group.category}] "${query}" -> intent=${result.intent}, action=${result.action}`,
        );
      }
    }
  }

  console.log("\n" + "=".repeat(72));
  console.log("Summary");
  console.log(`- Total queries: ${totalQueries}`);
  console.log(`- Intent checks passed: ${passedIntentChecks}/${totalQueries}`);
  console.log(`- Action checks passed: ${passedActionChecks}/${totalQueries}`);
  console.log(`- Full pass (intent + action): ${fullPasses}/${totalQueries}`);

  if (failures.length > 0) {
    console.log("\nFailures");
    for (const failure of failures) {
      console.log(`- ${failure}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log("\nAll checks passed.");
}

run().catch((error) => {
  console.error("Eval runner failed:", error);
  process.exitCode = 1;
});
